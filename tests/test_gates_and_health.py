"""Тесты гейтов и проверки здоровья разбора.

Три регрессионных теста фиксируют цифры, на которых стоят решения проекта: покрытие
объединения гейтов, атрибуцию владельца вызова и то, что нездоровый разбор гасит гейты.
Если эти цифры поедут от рефактора, тест обязан покраснеть раньше, чем это заметит сабмит.

Точность гейтов здесь НЕ проверяется, и это не упущение: на выданном сэмпле она не
определена — в страте «ответ с вызовом» всего два отрицательных примера, и оба недостижимы
ни для одного гейта. Аудит точности живёт в `scripts/audit_gates_on_trajectories.py`
и идёт по публичным траекториям, где отрицательные достижимы.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sverka.data import Example, load_examples
from sverka.gates import (
    MUTATING_PREFIXES,
    apply_gates,
    gate_multi_call,
    gate_placeholder_arg,
    gate_repeat_call,
    gate_unavailable_tool,
    load_known_user_tools,
)
from sverka.transcript import (
    ASSISTANT,
    ToolCall,
    ToolSchema,
    Transcript,
    Turn,
    parse_response,
    parse_transcript,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "valid.parquet"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="нет data/raw/valid.parquet")


@pytest.fixture(scope="module")
def examples() -> list[Example]:
    return load_examples(DATA)


def _mk_transcript(tools: list[str], history_calls: list[ToolCall] | None = None,
                   responses: tuple[tuple[str, str], ...] = ()) -> Transcript:
    turns = [Turn(role="SYSTEM", number=None, text="система")]
    if history_calls or responses:
        turns.append(
            Turn(role=ASSISTANT, number=1, text="", tool_calls=tuple(history_calls or ()),
                 tool_responses=responses)
        )
    return Transcript(
        system="система",
        policy="",
        instructions="",
        tools={t: ToolSchema(t, "", ()) for t in tools},
        turns=turns,
    )


def _call(name: str, raw: str = "{}", requestor: str = ASSISTANT) -> ToolCall:
    import json as _json

    try:
        args = _json.loads(raw)
    except Exception:
        args = {}
    return ToolCall(name=name, args=args if isinstance(args, dict) else {}, raw=raw, requestor=requestor)


# --- владелец вызова -------------------------------------------------------------

@needs_data
def test_call_ownership_is_attributed(examples):
    """Вызовы пользователя не должны приписываться агенту.

    В выданном сэмпле 393 вызова, из них 50 (12.7%) сделал пользователь: в telecom часть
    диагностики выполняет клиент на своём устройстве. Плоский поиск по промпту приписал бы
    их агенту и отравил бы любой признак «что агент уже делал».
    """
    agent = user = 0
    for e in examples:
        t = parse_transcript(e.prompt)
        agent += len(t.assistant_calls)
        user += len(t.user_calls)
    assert (agent, user) == (343, 50), f"атрибуция поехала: агент={agent} пользователь={user}"


# --- здоровье разбора ------------------------------------------------------------

@needs_data
def test_all_sample_parses_are_healthy(examples):
    for e in examples:
        t = parse_transcript(e.prompt)
        assert t.health is not None and t.health.ok, f"{e.id}: {t.health}"


def test_missing_tools_block_makes_parse_unhealthy():
    """Без [AVAILABLE TOOLS] разбор обязан признать себя нездоровым.

    Это не гигиена. При пустом словаре схем гейт «вызван недоступный инструмент»
    срабатывает не на пяти ходах, а на всех, где есть вызов, — смена формата даёт не
    тишину, а поток ложных срабатываний.
    """
    prompt = "⟦SYSTEM⟧\n<policy>правила</policy>\n⟦USER⟧\nпривет\n⟦ASSISTANT · ход 2⟧\nответ"
    t = parse_transcript(prompt)
    assert t.tools == {}
    assert t.health is not None and not t.health.ok
    assert any("схем инструментов" in r for r in t.health.reasons)


@needs_data
def test_unhealthy_parse_silences_all_gates(examples):
    e = examples[0]
    t = parse_transcript(e.prompt)
    turn = parse_response(e.response, strict=False)
    broken = Transcript(system=t.system, policy=t.policy, instructions="", tools={},
                        turns=t.turns, health=t.health)
    broken.tools = {}
    broken.health = parse_transcript(
        "⟦SYSTEM⟧\nбез схем\n⟦ASSISTANT · ход 1⟧\nх", strict=False
    ).health
    rep = apply_gates(broken, turn)
    assert rep.silenced and not rep.fired


def test_response_without_marker_silences_gates():
    tr = _mk_transcript(["book_reservation"])
    turn = parse_response("→ TOOL_CALL nonexistent_tool: {}", strict=False)
    assert turn.marker_found is False
    assert apply_gates(tr, turn).silenced


# --- гейт A ----------------------------------------------------------------------

def test_gate_a_fires_on_unknown_name():
    tr = _mk_transcript(["get_user_details", "book_reservation", "cancel_reservation",
                         "update_reservation_flights", "search_direct_flight"])
    turn = Turn(role=ASSISTANT, number=2, text="", tool_calls=(_call("check_network_status"),))
    hit = gate_unavailable_tool(tr, turn)
    assert hit is not None and "check_network_status" in hit.reason


def test_gate_a_does_not_fire_after_unlock():
    """Разблокированное имя недоступным не считается.

    Политика banking запрещает ПРЯМОЙ вызов discoverable-инструмента без разблокировки;
    после корректного `unlock_discoverable_agent_tool` вопрос открыт, и гейт туда не лезет.
    """
    unlock = _call("unlock_discoverable_agent_tool", '{"agent_tool_name": "close_bank_account_7392"}')
    tr = _mk_transcript(["a", "b", "c", "d", "unlock_discoverable_agent_tool"], [unlock])
    turn = Turn(role=ASSISTANT, number=3, text="", tool_calls=(_call("close_bank_account_7392"),))
    assert gate_unavailable_tool(tr, turn) is None


def test_gate_a_marks_user_side_tool_in_evidence():
    tr = _mk_transcript(["get_customer_by_phone", "b", "c", "d", "e"])
    turn = Turn(role=ASSISTANT, number=2, text="", tool_calls=(_call("check_sim_status"),))
    hit = gate_unavailable_tool(tr, turn, user_tools={"check_sim_status"})
    assert hit is not None
    assert any("устройстве пользователя" in ev for ev in hit.evidence)


# --- гейт B ----------------------------------------------------------------------

def test_gate_b_ignores_repeat_of_successful_call():
    """Повтор УСПЕШНОГО чтения — нормальное поведение, а не зацикливание.

    Замер на 1318 ходах решённых задач: широкая формулировка «повтор любого вызова» дала
    12 срабатываний, все на повторных get_*/search_* после успешного ответа.
    """
    prev = _call("get_order_details", '{"order_id": "#W1"}')
    tr = _mk_transcript(["get_order_details", "b", "c", "d", "e"], [prev],
                        responses=(("get_order_details", '{"status": "delivered"}'),))
    turn = Turn(role=ASSISTANT, number=3, text="", tool_calls=(prev,))
    assert gate_repeat_call(tr, turn) is None


def test_gate_b_fires_on_repeat_of_failed_call():
    prev = _call("find_user_id_by_name_zip", '{"zip": "80273"}')
    tr = _mk_transcript(["find_user_id_by_name_zip", "b", "c", "d", "e"], [prev],
                        responses=(("find_user_id_by_name_zip", "Error: User not found"),))
    turn = Turn(role=ASSISTANT, number=3, text="", tool_calls=(prev,))
    hit = gate_repeat_call(tr, turn)
    assert hit is not None and "find_user_id_by_name_zip" in hit.reason


def test_gate_b_does_not_collapse_two_broken_json_calls():
    """Два РАЗНЫХ вызова с неразобранным JSON не должны выглядеть повтором.

    При `JSONDecodeError` аргументы схлопываются в пустой словарь; сравнение по разобранным
    аргументам сделало бы их одинаковыми и дало бы ложное срабатывание.
    """
    a = _call("do_thing", '{"x": 1,,,}')
    b = _call("do_thing", '{"y": 2,,,}')
    assert not a.parsed_ok and not b.parsed_ok
    tr = _mk_transcript(["do_thing", "b", "c", "d", "e"], [a],
                        responses=(("do_thing", "Error: bad request"),))
    turn = Turn(role=ASSISTANT, number=3, text="", tool_calls=(b,))
    assert gate_repeat_call(tr, turn) is None


# --- гейты C и M -----------------------------------------------------------------

def test_gate_c_catches_placeholder_after_underscore():
    """`\\b` перед словом здесь не работает: подчёркивание — символ слова.

    Реальное срабатывание сэмпла — `"light_blue_account_id_placeholder"`.
    """
    turn = Turn(role=ASSISTANT, number=2, text="",
                tool_calls=(_call("close_account", '{"account_id": "light_blue_account_id_placeholder"}'),))
    assert gate_placeholder_arg(turn) is not None


def test_gate_c_ignores_normal_arguments():
    turn = Turn(role=ASSISTANT, number=2, text="",
                tool_calls=(_call("get_user", '{"user_id": "sofia_kim_7287"}'),))
    assert gate_placeholder_arg(turn) is None


def test_gate_m_mutating_needs_two_writes():
    one = Turn(role=ASSISTANT, number=2, text="",
               tool_calls=(_call("cancel_reservation"), _call("get_user_details")))
    two = Turn(role=ASSISTANT, number=2, text="",
               tool_calls=(_call("cancel_reservation"), _call("update_reservation_flights")))
    assert gate_multi_call(one, mutating_only=True) is None
    assert gate_multi_call(two, mutating_only=True) is not None


def test_transfer_is_not_a_mutation():
    """Перевод на оператора не меняет предметную базу — это передача разговора."""
    assert not any("transfer_to_human_agents".startswith(p) for p in MUTATING_PREFIXES)


def test_raw_multi_call_gate_is_off_by_default():
    """Сырой «два любых вызова» причинно не обоснован и по замеру избыточен."""
    tr = _mk_transcript(["get_a", "get_b", "c", "d", "e"])
    turn = Turn(role=ASSISTANT, number=2, text="", tool_calls=(_call("get_a"), _call("get_b")))
    assert not apply_gates(tr, turn).fired
    assert apply_gates(tr, turn, include_raw_multi=True).fired


# --- сводный регресс -------------------------------------------------------------

@needs_data
def test_gate_union_coverage_on_sample(examples):
    """Покрытие объединения гейтов на выданном сэмпле — зафиксированная цифра.

    Точность здесь намеренно не утверждается: достижимых отрицательных в страте ноль.
    """
    user_tools = load_known_user_tools()
    fired_pos = fired_neg = 0
    for e in examples:
        tr = parse_transcript(e.prompt, strict=False)
        turn = parse_response(e.response, strict=False)
        rep = apply_gates(tr, turn, user_tools=user_tools.get(e.domain, set()))
        if rep.fired:
            if e.label == 1:
                fired_pos += 1
            else:
                fired_neg += 1
    assert (fired_pos, fired_neg) == (11, 0), f"покрытие поехало: +{fired_pos} -{fired_neg}"


@needs_data
def test_user_tool_names_are_loaded_for_telecom():
    ut = load_known_user_tools()
    assert "check_network_status" in ut.get("telecom", set())
    assert "can_send_mms" in ut.get("telecom", set()), "имя есть в репозитории, но не в тексте политики"


# --- рекурсивный валидатор схемы --------------------------------------------------

from sverka.schema_check import validate_call, validate_turn  # noqa: E402
from sverka.transcript import ParamSpec  # noqa: E402


def _schema_with_nested() -> ToolSchema:
    """Схема book_reservation в том виде, в каком её отдаёт разбор [AVAILABLE TOOLS]."""
    return ToolSchema(
        "book_reservation",
        "Book a reservation.",
        (
            ParamSpec("user_id", "string", True),
            ParamSpec("cabin", "string", True, enum=("business", "economy", "basic_economy")),
            ParamSpec("flights", "array", True),
            ParamSpec("flight_number", "string", True, depth=1, parent="flights"),
            ParamSpec("date", "string", True, depth=1, parent="flights"),
            ParamSpec("passengers", "array", True),
            ParamSpec("first_name", "string", True, depth=1, parent="passengers"),
            ParamSpec("payment_methods", "array", True),
            ParamSpec("payment_id", "string", True, depth=1, parent="payment_methods"),
            ParamSpec("amount", "integer", True, depth=1, parent="payment_methods"),
            ParamSpec("total_baggages", "integer", True),
        ),
    )


def test_validator_catches_violation_inside_array_of_objects():
    """Ровно случай airline__23::t10, который плоская проверка не видит.

    Плоский валидатор дал 0 срабатываний на всём сэмпле: нарушение лежит внутри объекта
    внутри массива.
    """
    call = _call(
        "book_reservation",
        '{"user_id":"u1","cabin":"economy","flights":[{"flight_number":"HAT023","date":"2024-05-26"}],'
        '"passengers":[{"first_name":"Mohamed"}],'
        '"payment_methods":[{"id":"certificate_3765853","source":"certificate"}],"total_baggages":0}',
    )
    v = validate_call(call, _schema_with_nested())
    kinds = {(x.kind, x.path) for x in v}
    assert ("missing_required", "payment_methods[0].payment_id") in kinds
    assert ("missing_required", "payment_methods[0].amount") in kinds
    assert ("unknown_field", "payment_methods[0].id") in kinds


def test_validator_does_not_confuse_nested_namespaces():
    """`first_name` принадлежит passengers, а не payment_methods.

    Без поля `parent` оба лежали бы на depth=1 и были бы неразличимы — валидатор проверял
    бы поля не того объекта.
    """
    call = _call(
        "book_reservation",
        '{"user_id":"u1","cabin":"economy","flights":[{"flight_number":"H1","date":"d"}],'
        '"passengers":[{"first_name":"A"}],'
        '"payment_methods":[{"first_name":"A"}],"total_baggages":0}',
    )
    v = validate_call(call, _schema_with_nested())
    assert any(x.kind == "unknown_field" and x.path == "payment_methods[0].first_name" for x in v)
    assert not any(x.path == "passengers[0].first_name" for x in v)


def test_validator_flags_enum_and_type():
    call = _call(
        "book_reservation",
        '{"user_id":"u1","cabin":"first_class","flights":[{"flight_number":"H1","date":"d"}],'
        '"passengers":[{"first_name":"A"}],'
        '"payment_methods":[{"payment_id":"p","amount":"сто"}],"total_baggages":0}',
    )
    v = validate_call(call, _schema_with_nested())
    assert any(x.kind == "bad_enum" and x.path == "cabin" for x in v)
    assert any(x.kind == "wrong_type" and x.path == "payment_methods[0].amount" for x in v)


def test_validator_is_silent_when_schema_has_no_params():
    """Схемы из публичных траекторий содержат только имена — проверять там нечего,
    и молчание тут обязательно, иначе аудит утонет в выдуманных нарушениях."""
    call = _call("whatever", '{"a": 1}')
    assert validate_call(call, ToolSchema("whatever", "", ())) == []


def test_weak_and_strong_violations_are_separated():
    """Лишнее описательное поле при всех обязательных на месте — подсказка, не улика.

    На сэмпле `unknown_field` даёт верную метку на двух примерах, где организаторы назвали
    совсем другие причины. Считать это уликой значит повторить ошибку, за которую уже
    выброшен гейт «два любых вызова».
    """
    call = _call(
        "book_reservation",
        '{"user_id":"u1","cabin":"economy",'
        '"flights":[{"flight_number":"H1","date":"d","origin":"IAH","destination":"ORD"}],'
        '"passengers":[{"first_name":"A"}],'
        '"payment_methods":[{"payment_id":"p","amount":100}],"total_baggages":0}',
    )
    v = validate_call(call, _schema_with_nested())
    assert v and all(not x.strong for x in v), "лишние поля не должны считаться сильной уликой"


def test_validate_turn_skips_unknown_tools():
    """Вызов несуществующего инструмента — зона гейта A, дублировать его вердикт незачем."""
    turn = Turn(role=ASSISTANT, number=2, text="", tool_calls=(_call("no_such_tool", '{"x":1}'),))
    assert validate_turn(turn, {"book_reservation": _schema_with_nested()}) == []


# --- круговая проверка рендера корпуса ---------------------------------------------

import json as _json  # noqa: E402

from sverka.tau2 import iter_turns, load_tool_schemas, render_example  # noqa: E402


def _fake_trajectory(tmp_path):
    """Минимальная траектория в формате публичных файлов tau2-bench."""
    data = {
        "info": {
            "environment_info": {
                "domain_name": "airline",
                "policy": "# Airline Agent Policy\nThe current time is 2024-05-15 15:00:00 EST.",
                "tool_defs": None,
            }
        },
        "tasks": [],
        "simulations": [
            {
                "id": "s1",
                "task_id": "42",
                "trial": 0,
                "reward_info": {"reward": 1.0},
                "messages": [
                    {"role": "assistant", "turn_idx": 0, "content": "Hi!", "tool_calls": []},
                    {"role": "user", "turn_idx": 1, "content": "Cancel EHGLP3", "tool_calls": []},
                    {
                        "role": "assistant", "turn_idx": 2, "content": None,
                        "tool_calls": [{"id": "c1", "name": "get_reservation_details",
                                        "arguments": {"reservation_id": "EHGLP3"},
                                        "requestor": "assistant"}],
                    },
                    {"role": "tool", "turn_idx": 3, "id": "c1",
                     "content": '{"cabin": "economy"}', "requestor": "assistant"},
                    {
                        "role": "assistant", "turn_idx": 4, "content": "Отменяю.",
                        "tool_calls": [{"id": "c2", "name": "cancel_reservation",
                                        "arguments": {"reservation_id": "EHGLP3"},
                                        "requestor": "assistant"}],
                    },
                ],
            }
        ],
    }
    p = tmp_path / "traj.json"
    p.write_text(_json.dumps(data), encoding="utf-8")
    return p


def test_schemas_carry_nesting_and_required_flags():
    """Без вложенности рекурсивная проверка аргументов на добытых данных не работает.

    Две ловушки в исходниках tau2-bench: аргументы объявлены как `List[Passenger | dict]`
    (внутри подписки BinOp, а не имя), а поля Pydantic вида `Field(description=...)`
    выглядят как значение по умолчанию, хотя параметр обязателен.
    """
    s = load_tool_schemas("airline")
    assert "book_reservation" in s
    kids = {p.name: p.required for p in s["book_reservation"].children("payment_methods")}
    assert kids == {"payment_id": True, "amount": True}


def test_render_then_parse_round_trip(tmp_path):
    """Отрендерили → разобрали → получили то же самое.

    Собранный корпус обязан проходить через ровно тот же парсер, что и боевые данные,
    иначе он проверяет не то решение, которое поедет в сабмит.
    """
    turns = list(iter_turns(_fake_trajectory(tmp_path)))
    assert turns, "конвертер не выдал ни одного хода агента с вызовом"
    for tt in turns:
        prompt, response = render_example(tt)
        tr = parse_transcript(prompt, strict=False)
        tn = parse_response(response, strict=False)
        assert tr.health is not None and tr.health.ok, str(tr.health)
        assert set(tr.tools) == set(tt.transcript.tools)
        assert [c.name for c in tn.tool_calls] == [c.name for c in tt.turn.tool_calls]
        assert tn.marker_found


def test_render_round_trip_preserves_gate_verdict(tmp_path):
    user_tools = load_known_user_tools()
    for tt in iter_turns(_fake_trajectory(tmp_path)):
        before = apply_gates(tt.transcript, tt.turn, user_tools=user_tools.get(tt.domain, set()))
        prompt, response = render_example(tt)
        after = apply_gates(
            parse_transcript(prompt, strict=False),
            parse_response(response, strict=False),
            user_tools=user_tools.get(tt.domain, set()),
        )
        assert sorted(before.names) == sorted(after.names)


def test_user_calls_are_not_treated_as_agent_turns(tmp_path):
    """Ходы, где вызов сделал пользователь, в корпус агента попадать не должны."""
    p = tmp_path / "u.json"
    data = _json.loads(_fake_trajectory(tmp_path).read_text(encoding="utf-8"))
    data["simulations"][0]["messages"][4]["tool_calls"][0]["requestor"] = "user"
    p.write_text(_json.dumps(data), encoding="utf-8")
    names = [tt.turn_idx for tt in iter_turns(p)]
    assert 4 not in names

"""Конвертер публичных траекторий τ²-bench в наши структуры разбора.

Зачем. На выданных 46 примерах точность гейтов измерить нельзя: все они требуют вызова
инструмента в ответе, а в этой страте всего два отрицательных примера, и оба недостижимы
ни для одного гейта. Нужен набор, где отрицательные достижимы. Он существует и открыт:
полные траектории всех сабмитов лидерборда τ²-bench лежат в публичном бакете
`sierra-tau-bench-public` без авторизации.

Что даёт файл траектории (проверено на реальных данных):

    { "info": { "environment_info": { "domain_name": ..., "policy": <текст 7676 симв.>,
                                      "tool_defs": null },
                "git_commit": ... },
      "tasks": [ ... ],
      "simulations": [ { "id", "task_id", "trial", "termination_reason",
                         "reward_info": { "reward", "db_check", "action_checks", ... },
                         "messages": [ { "role", "turn_idx", "content",
                                         "tool_calls": [ {"id","name","arguments","requestor"} ] } ] } ] }

`tool_defs` в файлах пуст, поэтому схемы инструментов берутся из
`resources/tau2_tool_schemas.json` — выгрузки исходников tau2-bench (MIT), сделанной
разбором через ast: имя, тип, обязательность, вложенность на один уровень.

**Прокси-оракул.** Разметки на уровне хода в этих файлах нет (поле `review` у Sierra всегда
пустое). Но `reward == 1.0` означает, что задача решена: конечное состояние базы совпало
с эталоном и обязательные действия выполнены. Ходы таких симуляций — не гарантированно
чистые, но в подавляющем большинстве корректные. Срабатывание гейта на них поэтому читается
как вероятное ЛОЖНОЕ срабатывание. Это шумный оракул, и он честно назван прокси; его
единственная задача — дать гейтам шанс провалиться, которого выданный сэмпл не даёт.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sverka.transcript import (
    ASSISTANT,
    USER,
    ParamSpec,
    ParseHealth,
    ToolCall,
    ToolSchema,
    Transcript,
    Turn,
)

# Файл лежит ВНУТРИ пакета, а не в `model/`, и это не вкус.
# Платформа выполняет `pip install .`, после чего пакет живёт в site-packages,
# и путь вида `parents[2] / "model"` указывает мимо. Список инструментов тогда молча
# становится пустым, а гейт теряет часть обоснования — отказ тихий, заметный только
# по исчезнувшей строке в evidence.
_SCHEMAS_PATH = Path(__file__).resolve().parent / "resources" / "tau2_tool_schemas.json"


def _load_raw_schemas() -> dict:
    if not _SCHEMAS_PATH.exists():
        return {}
    return json.loads(_SCHEMAS_PATH.read_text(encoding="utf-8"))


def load_tool_schemas(domain: str) -> dict[str, ToolSchema]:
    """Схемы инструментов агента — с параметрами, обязательностью и вложенностью.

    Извлечены разбором исходников tau2-bench (MIT): в самих файлах траекторий поле
    `tool_defs` пусто, а без параметров рекурсивная проверка аргументов на добытых
    данных не работает вовсе.
    """
    out: dict[str, ToolSchema] = {}
    for name, t in (_load_raw_schemas().get(domain) or {}).items():
        if t.get("side") != "agent":
            continue
        params: list[ParamSpec] = []
        for p in t.get("params", []):
            params.append(
                ParamSpec(
                    name=p["name"],
                    type=p.get("type", ""),
                    required=bool(p.get("required")),
                    enum=tuple(p.get("enum") or ()),
                    description=p.get("description", ""),
                )
            )
            for c in p.get("children", []):
                params.append(
                    ParamSpec(
                        name=c["name"],
                        type=c.get("type", ""),
                        required=bool(c.get("required")),
                        enum=tuple(c.get("enum") or ()),
                        description=c.get("description", ""),
                        depth=1,
                        parent=p["name"],
                    )
                )
        out[name] = ToolSchema(name, t.get("description", ""), tuple(params))
    return out


@dataclass(frozen=True)
class TrajectoryTurn:
    """Один ход агента из публичной траектории, приведённый к нашим структурам."""

    source: str
    domain: str
    task_id: str
    trial: int
    turn_idx: int
    reward: float
    transcript: Transcript
    turn: Turn

    @property
    def uid(self) -> str:
        return f"{self.domain}__{self.task_id}::t{self.turn_idx}"


def load_agent_tool_names() -> dict[str, list[str]]:
    return {
        dom: sorted(n for n, t in tools.items() if t.get("side") == "agent")
        for dom, tools in _load_raw_schemas().items()
        if not dom.startswith("_") and isinstance(tools, dict)
    }


def _mk_call(tc: dict) -> ToolCall:
    args = tc.get("arguments")
    if not isinstance(args, dict):
        args = {}
    return ToolCall(
        name=str(tc.get("name") or ""),
        args=args,
        raw=json.dumps(args, sort_keys=True, ensure_ascii=False),
        requestor=USER if str(tc.get("requestor", "")).lower() == "user" else ASSISTANT,
    )


def _build_transcript(
    policy: str, schemas: dict[str, ToolSchema], history: list[dict], id_to_name: dict[str, str]
) -> Transcript:
    """Собрать транскрипт истории до проверяемого хода.

    Схемы приходят из выгрузки исходников tau2-bench, а не из файла траектории: в самих
    файлах `tool_defs` пуст. Благодаря этому на добытых данных работает и проверка
    аргументов, а не только проверка «такой инструмент вообще доступен».
    """
    tools = schemas
    turns: list[Turn] = []
    system = f"<policy>\n{policy}\n</policy>\n[AVAILABLE TOOLS]\n" + "\n".join(
        f"- {n} — {t.description}" for n, t in sorted(tools.items())
    )
    turns.append(Turn(role="SYSTEM", number=None, text=system))

    all_calls: list[ToolCall] = []
    for m in history:
        role = str(m.get("role", "")).lower()
        idx = m.get("turn_idx")
        if role == "tool":
            name = id_to_name.get(str(m.get("id")), "")
            turns.append(
                Turn(
                    role=USER if str(m.get("requestor", "")).lower() == "user" else ASSISTANT,
                    number=idx if isinstance(idx, int) else None,
                    text=str(m.get("content") or ""),
                    tool_responses=((name, str(m.get("content") or "")),),
                )
            )
            continue
        calls = tuple(_mk_call(tc) for tc in (m.get("tool_calls") or []))
        all_calls.extend(calls)
        turns.append(
            Turn(
                role=ASSISTANT if role == "assistant" else USER,
                number=idx if isinstance(idx, int) else None,
                text=str(m.get("content") or ""),
                tool_calls=calls,
            )
        )

    assistant = [c for c in all_calls if c.requestor == ASSISTANT]
    ratio = (
        sum(1 for c in assistant if c.name not in tools) / len(assistant) if assistant and tools else 0.0
    )
    health = ParseHealth(
        markers_found=len(turns),
        n_tools=len(tools),
        n_assistant_calls=len(assistant),
        unknown_call_ratio=ratio,
        ok=bool(tools),
        reasons=() if tools else ("нет списка инструментов для домена",),
    )
    return Transcript(
        system=system, policy=policy, instructions="", tools=tools, turns=turns, health=health
    )


def render_tool_schema(schema: ToolSchema) -> str:
    """Схема одного инструмента в том виде, в каком её печатают организаторы.

    Формат восстановлен по выданному сэмплу: `!` после типа означает обязательность,
    `[enum: a|b]` — допустимые значения, вложенное поле помечено `·` и большим отступом.
    """
    lines = [f"- {schema.name} — {schema.description}"]
    for p in schema.params:
        mark = "!" if p.required else ""
        enum = f" [enum: {'|'.join(p.enum)}]" if p.enum else ""
        desc = f" — {p.description}" if p.description else ""
        indent = "    " if p.depth == 0 else "        · "
        lines.append(f"{indent}{p.name}: {p.type}{mark}{enum}{desc}")
    return "\n".join(lines)


def render_example(tt: TrajectoryTurn) -> tuple[str, str]:
    """Ход траектории → пара (prompt, response) в формате организаторов.

    Нужно, чтобы собранный корпус проходил ровно через тот же парсер, что и боевые данные:
    иначе корпус проверял бы не то решение, которое поедет в сабмит. Круговая проверка
    «отрендерили → разобрали → получили те же вызовы» держит эти два формата вместе.
    """
    tr = tt.transcript
    tools_block = "\n".join(render_tool_schema(s) for _, s in sorted(tr.tools.items()))
    parts = [
        "⟦SYSTEM⟧",
        "<instructions>\nYou are a customer service agent that helps the user according to "
        "the <policy> provided below.\n</instructions>",
        f"<policy>\n{tr.policy}\n</policy>",
        f"[AVAILABLE TOOLS]\n{tools_block}",
    ]

    for t in tr.turns:
        if t.role == "SYSTEM":
            continue
        head = "⟦USER⟧" if t.role == USER else (
            f"⟦ASSISTANT · ход {t.number}⟧" if t.number is not None else "⟦ASSISTANT⟧"
        )
        body = [t.text.strip()] if (t.text or "").strip() else []
        for c in t.tool_calls:
            body.append(f"\t→ TOOL_CALL {c.name}: {c.raw}")
        for name, res in t.tool_responses:
            body.append(f"\t← TOOL_RESPONSE {name}: {res}")
        parts.append(head + ("\n" + "\n".join(body) if body else ""))

    resp_body = [tt.turn.text.strip()] if (tt.turn.text or "").strip() else []
    for c in tt.turn.tool_calls:
        resp_body.append(f"\t→ TOOL_CALL {c.name}: {c.raw}")
    response = f"⟦ASSISTANT · ход {tt.turn_idx}⟧" + (
        "\n" + "\n".join(resp_body) if resp_body else ""
    )
    return "\n\n".join(parts), response


def iter_turns(
    path: str | Path,
    *,
    only_reward: float | None = None,
    require_tool_calls: bool = True,
    max_sims: int | None = None,
) -> Iterator[TrajectoryTurn]:
    """Пройти файл траекторий и выдать ходы агента, приведённые к нашим структурам.

    `only_reward=1.0` оставляет лишь симуляции, где задача решена — это и есть прокси-оракул
    «ход, скорее всего, корректен».
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    info = data.get("info", {}) or {}
    env = info.get("environment_info", {}) or {}
    domain = str(env.get("domain_name") or "")
    policy = str(env.get("policy") or "")
    schemas = load_tool_schemas(domain)

    sims = data.get("simulations") or []
    if max_sims is not None:
        sims = sims[:max_sims]

    for sim in sims:
        reward = float((sim.get("reward_info") or {}).get("reward") or 0.0)
        if only_reward is not None and reward != only_reward:
            continue
        messages = sim.get("messages") or []

        id_to_name: dict[str, str] = {}
        for m in messages:
            for tc in m.get("tool_calls") or []:
                if tc.get("id"):
                    id_to_name[str(tc["id"])] = str(tc.get("name") or "")

        for i, m in enumerate(messages):
            if str(m.get("role", "")).lower() != "assistant":
                continue
            calls = tuple(_mk_call(tc) for tc in (m.get("tool_calls") or []))
            if require_tool_calls and not calls:
                continue
            if any(c.requestor == USER for c in calls):
                continue  # ход пользователя, а не агента
            tr = _build_transcript(policy, schemas, messages[:i], id_to_name)
            turn = Turn(
                role=ASSISTANT,
                number=m.get("turn_idx") if isinstance(m.get("turn_idx"), int) else None,
                text=str(m.get("content") or ""),
                tool_calls=calls,
                marker_found=True,
            )
            yield TrajectoryTurn(
                source=path.name,
                domain=domain,
                task_id=str(sim.get("task_id") or sim.get("id") or "?"),
                trial=int(sim.get("trial") or 0),
                turn_idx=int(m.get("turn_idx") or i),
                reward=reward,
                transcript=tr,
                turn=turn,
            )

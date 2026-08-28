"""Детерминированные гейты — дешёвые проверки, не требующие модели.

⚠️ **Ни один гейт здесь НЕ имеет права самостоятельно ставить метку.** На выданном сэмпле
их точность не измерена и измерена быть не может: все гейты требуют вызова инструмента в
ответе, то есть живут внутри страты «ответ с вызовом», а в ней 23 примера при всего двух
отрицательных — и оба это одиночный `transfer_to_human_agents` с валидным именем и JSON,
на который ни один гейт сработать не способен. Достижимых отрицательных ноль, значит
наблюдение «ноль ложных срабатываний» не измеряет ничего: при случайном выборе 12 строк
из этой страты вероятность не задеть отрицательный равна 0.217.

Поэтому гейты используются как **признаки** — вход для судьи и материал для аудита на
собственном корпусе. Право ставить метку выдаётся отдельно и только после того, как
точность померена на траекториях, где отрицательные достижимы.

Второе ограничение, столь же важное: **при нездоровом разборе все гейты молчат.** Если
блок `[AVAILABLE TOOLS]` не распознан, словарь схем пуст, и гейт «вызван недоступный
инструмент» срабатывает не на пяти ходах, а на всех, где вообще есть вызов. Смена формата
даёт не тишину, а поток ложных срабатываний, поэтому `Gate.apply` начинается с проверки
`transcript.health.ok`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from sverka.transcript import ToolCall, Transcript, Turn

# Строки-заглушки: агент подставил очевидный плейсхолдер вместо настоящего значения.
# Список намеренно узкий — это должно быть самоочевидно, а не «похоже на выдумку».
#
# ⚠️ Границы слова `\b` здесь ставить нельзя: реальное срабатывание сэмпла —
# `"light_blue_account_id_placeholder"`, где перед словом стоит подчёркивание, а оно
# считается символом слова, и `\bplaceholder\b` не совпадает. Ошибка тихая: гейт просто
# перестаёт срабатывать, и это видно только по счётчику.
PLACEHOLDER_RE = re.compile(
    r"placeholder|example_[a-z0-9_]*|\bTODO\b|\bFIXME\b|<[a-z_]{3,}>|\bxxx+\b|dummy|your_[a-z_]+_here",
    re.I,
)

# Префиксы инструментов, меняющих состояние. Нужны для «мутирующего» варианта гейта M:
# сырой вариант («любые два вызова за ход») с разметкой причинно не связан, а вот два
# записывающих действия подряд без подтверждения — это уже поведение, которое политики
# доменов запрещают явно.
#
# `transfer_*` в список НЕ входит: перевод на оператора не меняет состояние предметной базы,
# это передача разговора. Аудит на траекториях поймал ровно один такой случай
# (`update_reservation_flights` + `transfer_to_human_agents`), и засчитывать его как
# «две записи за ход» было бы неверно по существу.
MUTATING_PREFIXES = (
    "book_", "cancel_", "update_", "modify_", "return_", "exchange_", "refund_",
    "send_", "close_", "approve_", "resume_", "suspend_", "log_",
    "set_", "enable_", "disable_", "delete_", "remove_", "create_", "add_",
)

# Файл лежит ВНУТРИ пакета, а не в `model/`, и это не вкус.
# Платформа выполняет `pip install .`, после чего пакет живёт в site-packages,
# и путь вида `parents[2] / "model"` указывает мимо. Список пользовательских
# инструментов тогда молча становится пустым, а гейт теряет часть обоснования —
# отказ тихий, заметный только по исчезнувшей строке в evidence.
_TOOL_NAMES_PATH = Path(__file__).resolve().parent / "resources" / "tau2_tool_names.json"


def load_known_user_tools() -> dict[str, set[str]]:
    """Имена инструментов, которые в τ²-bench выполняет ПОЛЬЗОВАТЕЛЬ, а не агент.

    Нужны, чтобы отличать два разных дефекта: агент выдумал имя (его нет нигде) или агент
    попытался выполнить действие, которое по политике должен выполнить клиент на своём
    устройстве. Второе — самый частый случай в telecom, и рубрика организаторов размечает
    его как галлюцинацию.

    Список берётся из репозитория tau2-bench (MIT). Из текста политики он не восстанавливается
    полностью: `can_send_mms` и `check_payment_request` вызываются пользователем, но в
    нумерованный перечень политики не входят.
    """
    if not _TOOL_NAMES_PATH.exists():
        return {}
    data = json.loads(_TOOL_NAMES_PATH.read_text(encoding="utf-8"))
    return {
        dom: set(v.get("user", []))
        for dom, v in data.items()
        if not dom.startswith("_") and isinstance(v, dict)
    }


@dataclass(frozen=True)
class GateHit:
    """Срабатывание одного гейта с обоснованием.

    `evidence` обязателен: без привязки к куску контекста нельзя отличить «сработало
    правильно» от «совпало случайно».
    """

    gate: str
    reason: str
    evidence: tuple[str, ...] = ()


@dataclass
class GateReport:
    hits: list[GateHit] = field(default_factory=list)
    silenced: bool = False
    silence_reason: str = ""

    @property
    def fired(self) -> bool:
        return bool(self.hits)

    @property
    def names(self) -> list[str]:
        return [h.gate for h in self.hits]

    def __str__(self) -> str:
        if self.silenced:
            return f"гейты отключены: {self.silence_reason}"
        if not self.hits:
            return "ни один гейт не сработал"
        return "; ".join(f"{h.gate}: {h.reason}" for h in self.hits)


def _call_key(c: ToolCall) -> tuple[str, str]:
    """Ключ для сравнения вызовов.

    Сравнивается СЫРАЯ строка аргументов, а не разобранный словарь. При неудачном разборе
    JSON `args` становится пустым словарём, и два разных битых вызова одного инструмента
    выглядели бы одинаково — гейт повторов дал бы ложное срабатывание там, где его нет.
    """
    return (c.name, " ".join(c.raw.split()))


def gate_unavailable_tool(
    tr: Transcript, turn: Turn, *, user_tools: set[str] | None = None
) -> GateHit | None:
    """A — агент вызвал инструмент, которого нет в его `[AVAILABLE TOOLS]`.

    Уточнение: имя, разблокированное ранее через `unlock_discoverable_agent_tool`, не
    считается недоступным. Политика banking запрещает ПРЯМОЙ вызов discoverable-инструмента
    без разблокировки, но после неё вопрос открыт, и гейт туда не лезет.

    ⚠️ На выданном сэмпле это уточнение непроверяемо: строки, где оно обязано погасить
    срабатывание, в наборе нет — восемь discoverable-вызовов лежат в истории, а не в
    проверяемом ответе. Патч подтверждён только тем, что ничего не сломал.
    """
    if not tr.tools:
        return None
    unlocked = tr.unlocked_tools()
    bad = [c.name for c in turn.tool_calls if c.name not in tr.tools and c.name not in unlocked]
    if not bad:
        return None
    ev = [f"в [AVAILABLE TOOLS] {len(tr.tools)} инструментов, среди них нет: {sorted(set(bad))}"]
    if user_tools:
        side = sorted({n for n in bad if n in user_tools})
        if side:
            ev.append(f"это действия на устройстве пользователя, агент их выполнять не может: {side}")
    return GateHit("A_unavailable_tool", f"вызваны недоступные инструменты: {sorted(set(bad))}", tuple(ev))


def gate_multi_call(turn: Turn, *, mutating_only: bool = False) -> GateHit | None:
    """M — два и более вызова инструментов за один ход.

    ⚠️ Сырой вариант с разметкой причинно НЕ связан, и это замерено: ни одно из 23
    объяснений выданного сэмпла не упоминает множественность вызовов, а сам многовызов
    встречается в 53 ходах истории (23% всех ходов с вызовами) и в 9 отрицательных примерах.
    Признак коррелирует с меткой, но объясняется не правилом, а сложностью хода.

    `mutating_only=True` сужает до подслучая, который политики действительно запрещают:
    два и более действия, меняющих состояние, в одном ходу.

    `[замер]` Сырой вариант оказался ещё и **избыточным**: на выданном сэмпле объединение
    без него даёт ровно те же 12 срабатываний, что и с ним (F1 = 0.686, R = 0.522). То есть
    причинно необоснованный признак выбрасывается бесплатно, и поэтому он выключен по
    умолчанию в `apply_gates`.
    """
    calls = list(turn.tool_calls)
    if mutating_only:
        calls = [c for c in calls if c.name.startswith(MUTATING_PREFIXES)]
    if len(calls) < 2:
        return None
    names = [c.name for c in calls]
    kind = "мутирующих вызова" if mutating_only else "вызова"
    return GateHit(
        "M_multi_mutating" if mutating_only else "M_multi_call",
        f"{len(calls)} {kind} за один ход: {names}",
        (f"вызовы хода: {[c.name for c in turn.tool_calls]}",),
    )


ERROR_RESPONSE_RE = re.compile(r"^\s*(?:error\b|exception\b)|not found|is not available|failed", re.I)


def _failed_tool_names(tr: Transcript) -> set[str]:
    """Инструменты, чей вызов хотя бы раз вернул ошибку."""
    out: set[str] = set()
    for t in tr.turns:
        for name, body in t.tool_responses:
            head = (body or "")[:200]
            if ERROR_RESPONSE_RE.search(head):
                out.add(name)
    return out


def gate_repeat_call(tr: Transcript, turn: Turn) -> GateHit | None:
    """B — дословный повтор вызова, который **ранее уже провалился**.

    ⚠️ Гейт сужен по итогам аудита на публичных траекториях (T7), и это тот случай, когда
    замер опроверг первоначальную формулировку.

    Было: «повтор любого ранее сделанного вызова». `[замер]` На 1318 ходах решённых задач
    (`reward == 1.0`) такая формулировка дала 12 срабатываний, и все — повторы УСПЕШНЫХ
    read-only вызовов: `get_user_details`, `get_order_details`, `get_product_details`,
    `search_onestop_flight`. Перечитать состояние после изменения — нормальное поведение
    агента, а не галлюцинация. Это был настоящий генератор ложных срабатываний.

    Стало: повтор засчитывается, только если этот инструмент ранее уже возвращал ошибку.
    `[замер]` На выданном сэмпле все три срабатывания были именно такими и сохраняются:
    `find_user_id_by_name_zip` после «Error: User not found», `check_network_status` и
    `run_speed_test` после «Error: Tool ... not found». Смысл гейта — зацикливание:
    агент повторяет то, что уже не сработало, ничего не изменив.

    Сравниваются только вызовы АГЕНТА и только с разобранным JSON: при неудачном разборе
    аргументы схлопываются в пустой словарь, и два разных битых вызова выглядели бы
    одинаково.
    """
    failed = _failed_tool_names(tr)
    if not failed:
        return None
    prev = {_call_key(c) for c in tr.assistant_calls if c.parsed_ok}
    repeats = [
        c.name
        for c in turn.tool_calls
        if c.parsed_ok and c.name in failed and _call_key(c) in prev
    ]
    if not repeats:
        return None
    return GateHit(
        "B_repeat_failed_call",
        f"повтор ранее провалившегося вызова: {sorted(set(repeats))}",
        (
            f"инструменты с ошибкой в истории: {sorted(failed)[:6]}",
            f"в истории агента {len(prev)} уникальных вызовов",
        ),
    )


def gate_placeholder_arg(turn: Turn) -> GateHit | None:
    """C — в аргументах вызова стоит очевидная строка-заглушка вместо значения."""
    found: list[tuple[str, str]] = []
    evidence: list[str] = []
    for c in turn.tool_calls:
        m = PLACEHOLDER_RE.search(c.raw)
        if m:
            found.append((c.name, m.group(0)))
            evidence.append(f"{c.name}: {c.raw[:160]}")
    if not found:
        return None
    return GateHit("C_placeholder_arg", f"заглушка в аргументах: {found}", tuple(evidence[:3]))


def gate_schema_violation(tr: Transcript, turn: Turn, *, strong_only: bool = True) -> GateHit | None:
    """D — аргументы вызова не соответствуют схеме инструмента.

    Проверка рекурсивная: `[замер]` плоская не сработала ни разу, потому что единственное
    схемное нарушение сэмпла лежит внутри объекта внутри массива.

    По умолчанию учитываются только сильные нарушения — отсутствие обязательного поля,
    неверный тип, значение вне `enum`. Лишнее поле при полном комплекте обязательных
    (`origin`, `destination`, `price` внутри `flights[]`) считается подсказкой, а не
    уликой: на сэмпле оно даёт верную метку по причине, которой организаторы не называли.
    """
    from sverka.schema_check import validate_turn

    viols = validate_turn(turn, tr.tools)
    if strong_only:
        viols = [(n, v) for n, v in viols if v.strong]
    if not viols:
        return None
    return GateHit(
        "D_schema_violation",
        f"аргументы не соответствуют схеме: {[f'{n}:{v.kind}@{v.path}' for n, v in viols[:4]]}",
        tuple(f"{n}: {v}" for _, v in viols[:3] for n in [viols[0][0]]),
    )


def apply_gates(
    tr: Transcript,
    turn: Turn,
    *,
    user_tools: set[str] | None = None,
    include_raw_multi: bool = False,
    include_weak_schema: bool = False,
) -> GateReport:
    """Прогнать все гейты по разобранному примеру.

    При нездоровом разборе гейты молчат: это не осторожность, а необходимость. Пустой
    словарь схем превращает гейт A из «сработал на пяти ходах» в «сработал на всех».
    """
    rep = GateReport()

    if not turn.marker_found:
        rep.silenced = True
        rep.silence_reason = "в ответе не найден маркер хода — формат отличается от ожидаемого"
        return rep
    if tr.health is not None and not tr.health.ok:
        rep.silenced = True
        rep.silence_reason = str(tr.health)
        return rep

    for hit in (
        gate_unavailable_tool(tr, turn, user_tools=user_tools),
        gate_repeat_call(tr, turn),
        gate_placeholder_arg(turn),
        gate_schema_violation(tr, turn, strong_only=not include_weak_schema),
        gate_multi_call(turn, mutating_only=True),
        gate_multi_call(turn, mutating_only=False) if include_raw_multi else None,
    ):
        if hit is not None:
            rep.hits.append(hit)
    return rep

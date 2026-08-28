"""Разбор транскрипта.

⚠️ Формат восстановлен по фактическому содержимому `data/raw/valid.parquet` (27.08.2026),
а НЕ по описанию задачи — они расходятся. Документировано три несовместимых описания:

1. **фактическое** (вкладка Data и сам файл): блоки `⟦SYSTEM⟧`, `⟦USER⟧`,
   `⟦ASSISTANT · ход N⟧`, а вызовы — строками внутри тела хода:
   `→ TOOL_CALL имя: {json}` и `← TOOL_RESPONSE имя: результат`;
2. **маркерное** (вкладка Overview): `⟦ASSISTANT_TOOL_CALL name="X"⟧`,
   `⟦USER_TOOL_CALL name="X"⟧`, `⟦TOOL_RESULT name="X" requestor="assistant"⟧`;
3. **единый текст** с тегами `<prompt>…</prompt>` / `<response>…</response>`.

Вместо трёх веток кода здесь один нормализующий проход: текст режется по любым маркерам
`⟦…⟧`, каждый блок получает владельца, а вызовы внутри тела приписываются владельцу блока.
Два из трёх форматов в выданных данных не встречались ни разу, поэтому отдельные ветки под
них были бы кодом без единого наблюдения.

## Владелец вызова — не деталь, а условие корректности

`[замер]` Из 393 вызовов в промптах выданного сэмпла **50 (12.7%) принадлежат пользователю**,
а не агенту (telecom 45, banking 5): в этих доменах часть диагностики выполняет сам клиент
на своём устройстве. Плоский поиск `TOOL_CALL` по всему промпту приписывает агенту 13% чужих
действий и отравляет любой признак вида «что агент уже делал».

## Здоровье разбора — защита от тихой катастрофы

`parse_tool_schemas` возвращает пустой словарь, если не нашла `[AVAILABLE TOOLS]`, а нестрогий
разбор при отсутствии маркеров возвращает пустой транскрипт. Исключения нет, вызывающий код
его не заметит. При пустом словаре схем детектор «вызван инструмент вне списка» срабатывает
**на всех** ходах с вызовом, а не на пяти — то есть смена формата даёт не тишину, а обвал
точности. Поэтому разбор всегда сопровождается объектом `ParseHealth`, и потребитель обязан
гасить проверки, когда `health.ok` ложно.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

LB, RB = "⟦", "⟧"
_MARKER_RE = re.compile(rf"{LB}([^{RB}]{{0,120}}){RB}")
_TURN_NO_RE = re.compile(r"ход\s+(\d+)")
_TOOL_CALL_RE = re.compile(r"[→>]\s*TOOL_CALL\s+([A-Za-z0-9_.\-]+)\s*:\s*(\{.*?\})\s*(?=\n|$)", re.S)
_TOOL_RESP_RE = re.compile(
    r"[←<]\s*TOOL_RESPONSE\s+([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*:\s*(.*?)\s*(?=\n\s*[→←>]|\n\s*\n|$)",
    re.S,
)
_NAME_ATTR_RE = re.compile(r'name\s*=\s*"([^"]+)"')
_REQUESTOR_ATTR_RE = re.compile(r'requestor\s*=\s*"([^"]+)"')
_TOOLS_HEADER = "[AVAILABLE TOOLS]"

ASSISTANT, USER, SYSTEM = "ASSISTANT", "USER", "SYSTEM"

# Пороги здоровья разбора взяты из замера, а не назначены на глаз.
#
# `MIN_TOOLS_HEALTHY`: в выданном сэмпле в [AVAILABLE TOOLS] 13–17 инструментов при
# медиане 14.9, так что пять — заведомо ниже любого настоящего блока схем.
#
# `MAX_UNKNOWN_CALL_RATIO` разводит два состояния, которые иначе слиплись бы:
#   • настоящая галлюцинация — агент зовёт недоступный ему инструмент. Максимум на сэмпле
#     0.50 (telecom, где часть диагностики выполняет пользователь), медиана 0.00;
#   • сломанный разбор схем — тогда НИ ОДНО имя не найдётся, и доля равна ровно 1.00
#     (это случилось бы на 40 из 46 примеров, где у агента вообще есть вызовы).
# Порог 0.8 лежит между наблюдаемым максимумом и единицей. Занизить его нельзя: гейт
# «вызван недоступный инструмент» замолчал бы ровно на тех примерах, ради которых нужен.
MIN_TOOLS_HEALTHY = 5
MAX_UNKNOWN_CALL_RATIO = 0.8


class TranscriptError(RuntimeError):
    """Транскрипт не разобрался. Падаем громко и с текстом, а не молча."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    raw: str
    requestor: str = ASSISTANT  # кто сделал вызов: ASSISTANT или USER

    @property
    def parsed_ok(self) -> bool:
        """JSON аргументов разобрался (пустой объект тоже валиден).

        Отличать «аргументов нет» от «аргументы не разобрались» обязательно: при неудачном
        разборе `args` становится пустым словарём, и два РАЗНЫХ битых вызова одного
        инструмента выглядят одинаково. Детектор повторов обязан сравнивать `raw`.
        """
        stripped = self.raw.strip()
        if stripped in ("{}", ""):
            return True
        return bool(self.args)


@dataclass(frozen=True)
class ParamSpec:
    """Один параметр в схеме инструмента из [AVAILABLE TOOLS].

    `parent` обязателен для рекурсивной проверки аргументов: без него `payment_id` внутри
    `payment_methods` неотличим от `first_name` внутри `passengers` — у обоих `depth = 1`.
    """

    name: str
    type: str
    required: bool
    enum: tuple[str, ...] = ()
    description: str = ""
    depth: int = 0
    parent: str | None = None


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    params: tuple[ParamSpec, ...]

    @property
    def required_names(self) -> set[str]:
        return {p.name for p in self.params if p.required and p.depth == 0}

    @property
    def known_names(self) -> set[str]:
        return {p.name for p in self.params if p.depth == 0}

    def children(self, parent: str) -> tuple[ParamSpec, ...]:
        """Вложенные поля указанного параметра-контейнера."""
        return tuple(p for p in self.params if p.parent == parent)


@dataclass(frozen=True)
class Turn:
    role: str  # SYSTEM | USER | ASSISTANT
    number: int | None  # номер хода (у приветствия и реплик пользователя None)
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_responses: tuple[tuple[str, str], ...] = ()
    marker_found: bool = True  # был ли найден маркер ⟦…⟧, или разбор шёл вслепую


@dataclass(frozen=True)
class ParseHealth:
    """Признак того, что разбор дал осмысленный результат, а не тихо пустой.

    `ok = False` означает: формат отличается от ожидаемого, и детерминированные проверки
    обязаны молчать — иначе они дадут не тишину, а поток ложных срабатываний.
    """

    markers_found: int
    n_tools: int
    n_assistant_calls: int
    unknown_call_ratio: float
    ok: bool
    reasons: tuple[str, ...] = ()

    def __str__(self) -> str:
        state = "здоров" if self.ok else "НЕЗДОРОВ: " + "; ".join(self.reasons)
        return (
            f"разбор {state} (маркеров={self.markers_found}, схем={self.n_tools}, "
            f"вызовов агента={self.n_assistant_calls}, доля неизвестных={self.unknown_call_ratio:.2f})"
        )


@dataclass
class Transcript:
    system: str
    policy: str
    instructions: str
    tools: dict[str, ToolSchema] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    health: ParseHealth | None = None

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == ASSISTANT]

    @property
    def assistant_calls(self) -> list[ToolCall]:
        """Вызовы, сделанные именно агентом. Основа любого признака «что агент уже делал»."""
        return [c for t in self.turns for c in t.tool_calls if c.requestor == ASSISTANT]

    @property
    def user_calls(self) -> list[ToolCall]:
        return [c for t in self.turns for c in t.tool_calls if c.requestor == USER]

    def turns_before(self, number: int) -> list[Turn]:
        """Ходы до указанного номера — история, доступная агенту на проверяемом ходу."""
        out = []
        for t in self.turns:
            if t.number is not None and t.number >= number:
                break
            out.append(t)
        return out

    def unlocked_tools(self) -> set[str]:
        """Имена, разблокированные через `unlock_discoverable_agent_tool`.

        Нужны детектору недоступных инструментов: прямой вызов discoverable-имени без
        предварительной разблокировки политика запрещает, а после разблокировки — вопрос
        открытый, и такой вызов детектор трогать не должен.
        """
        out: set[str] = set()
        for t in self.turns:
            for c in t.tool_calls:
                if c.name == "unlock_discoverable_agent_tool":
                    v = c.args.get("agent_tool_name")
                    if isinstance(v, str):
                        out.add(v)
            for name, body in t.tool_responses:
                if name == "unlock_discoverable_agent_tool":
                    for m in re.finditer(r'"?(?:agent_tool_name|name)"?\s*[:=]\s*"([A-Za-z0-9_]+)"', body):
                        out.add(m.group(1))
        return out


def strip_document_tags(text: str) -> tuple[str, str | None]:
    """Снять обёртку `<prompt>…</prompt>` / `<response>…</response>`, если она есть.

    Третий документированный формат подаёт пару одним текстом. В выданных данных не
    встречался ни разу, но проверяется первым делом на реальном тесте.
    """
    p = re.search(r"<prompt>(.*?)</prompt>", text, re.S)
    r = re.search(r"<response>(.*?)</response>", text, re.S)
    if p:
        return p.group(1).strip(), (r.group(1).strip() if r else None)
    return text, None


def _extract_block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else ""


def parse_tool_schemas(system_text: str) -> dict[str, ToolSchema]:
    """Разобрать блок [AVAILABLE TOOLS] в словарь схем.

    Формат строки инструмента:  `- name — description`
    Формат параметра:           `    name: type! [enum: a|b] — description`
    Вложенные поля помечены `·` и большим отступом; `!` после типа означает обязательность.
    Каждому вложенному полю проставляется `parent` — последний параметр меньшей глубины.
    """
    i = system_text.find(_TOOLS_HEADER)
    if i < 0:
        return {}
    body = system_text[i + len(_TOOLS_HEADER):]

    tools: dict[str, ToolSchema] = {}
    cur_name: str | None = None
    cur_desc = ""
    cur_params: list[ParamSpec] = []
    last_top: str | None = None

    def flush() -> None:
        if cur_name:
            tools[cur_name] = ToolSchema(cur_name, cur_desc, tuple(cur_params))

    for line in body.splitlines():
        if not line.strip():
            continue
        m_tool = re.match(r"^-\s+([A-Za-z0-9_.\-]+)\s*[—-]\s*(.*)$", line.strip())
        if m_tool and not line.startswith((" ", "\t")):
            flush()
            cur_name, cur_desc, cur_params = m_tool.group(1), m_tool.group(2).strip(), []
            last_top = None
            continue
        if cur_name is None:
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip().lstrip("·").strip()
        m_par = re.match(
            r"^([A-Za-z0-9_]+)\s*:\s*([A-Za-z_]+)(!?)\s*(?:\[enum:\s*([^\]]+)\])?\s*(?:[—-]\s*(.*))?$",
            stripped,
        )
        if not m_par:
            continue
        depth = 0 if indent <= 4 else 1
        name = m_par.group(1)
        cur_params.append(
            ParamSpec(
                name=name,
                type=m_par.group(2),
                required=m_par.group(3) == "!",
                enum=tuple(x.strip() for x in m_par.group(4).split("|")) if m_par.group(4) else (),
                description=(m_par.group(5) or "").strip(),
                depth=depth,
                parent=None if depth == 0 else last_top,
            )
        )
        if depth == 0:
            last_top = name
    flush()
    return tools


def _parse_inline_calls(
    body: str, requestor: str
) -> tuple[tuple[ToolCall, ...], tuple[tuple[str, str], ...]]:
    """Вызовы и ответы, лежащие строками внутри тела блока (фактический формат)."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(body):
        raw = m.group(2)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        calls.append(
            ToolCall(
                name=m.group(1),
                args=parsed if isinstance(parsed, dict) else {},
                raw=raw,
                requestor=requestor,
            )
        )
    resps = tuple((m.group(1), m.group(2)) for m in _TOOL_RESP_RE.finditer(body))
    return tuple(calls), resps


def _classify(head: str) -> tuple[str, str | None, int | None, str | None]:
    """Разобрать содержимое маркера в (роль, вид, номер хода, имя инструмента).

    Вид: None — обычный блок роли; "call" — маркерный вызов; "result" — маркерный результат.
    """
    upper = head.upper()
    num_m = _TURN_NO_RE.search(head)
    number = int(num_m.group(1)) if num_m else None
    name_m = _NAME_ATTR_RE.search(head)
    name = name_m.group(1) if name_m else None

    if upper.startswith("ASSISTANT_TOOL_CALL"):
        return ASSISTANT, "call", number, name
    if upper.startswith("USER_TOOL_CALL"):
        return USER, "call", number, name
    if upper.startswith("TOOL_RESULT"):
        req = _REQUESTOR_ATTR_RE.search(head)
        who = (req.group(1).upper() if req else ASSISTANT)
        return (USER if who == USER else ASSISTANT), "result", number, name
    role = upper.split("·")[0].strip().split()[0] if upper.strip() else ""
    if role.startswith("SYSTEM"):
        return SYSTEM, None, None, None
    if role.startswith("USER"):
        return USER, None, number, None
    if role.startswith("ASSISTANT"):
        return ASSISTANT, None, number, None
    return role or "UNKNOWN", None, number, None


def _health(markers: int, tools: dict, calls: list[ToolCall]) -> ParseHealth:
    reasons: list[str] = []
    if markers == 0:
        reasons.append("не найдено ни одного маркера ⟦…⟧")
    n_tools = len(tools)
    if n_tools < MIN_TOOLS_HEALTHY:
        reasons.append(f"схем инструментов {n_tools} < {MIN_TOOLS_HEALTHY}")
    assistant = [c for c in calls if c.requestor == ASSISTANT]
    ratio = 0.0
    if assistant and tools:
        ratio = sum(1 for c in assistant if c.name not in tools) / len(assistant)
        if ratio > MAX_UNKNOWN_CALL_RATIO:
            reasons.append(f"доля вызовов агента вне схемы {ratio:.2f} > {MAX_UNKNOWN_CALL_RATIO}")
    return ParseHealth(
        markers_found=markers,
        n_tools=n_tools,
        n_assistant_calls=len(assistant),
        unknown_call_ratio=ratio,
        ok=not reasons,
        reasons=tuple(reasons),
    )


def parse_transcript(prompt: str, *, strict: bool = True) -> Transcript:
    """Разобрать поле `prompt` в структуру.

    strict=True (по умолчанию): нет маркеров или нет системного блока → TranscriptError.
    В боевом `predict.py` вызывать со strict=False и смотреть на `transcript.health`:
    молчаливый пустой разбор опаснее падения, потому что съедает сабмит и выдаёт мусор.
    """
    prompt, _ = strip_document_tags(prompt)
    markers = list(_MARKER_RE.finditer(prompt))
    if not markers:
        if strict:
            raise TranscriptError(
                "в prompt не найдено ни одного маркера ⟦…⟧ — формат отличается от ожидаемого; "
                f"первые 200 символов: {prompt[:200]!r}"
            )
        return Transcript(
            system="",
            policy="",
            instructions="",
            turns=[],
            health=_health(0, {}, []),
        )

    turns: list[Turn] = []
    system_text = ""
    all_calls: list[ToolCall] = []

    for idx, m in enumerate(markers):
        head = m.group(1).strip()
        start = m.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(prompt)
        body = prompt[start:end].strip("\n")
        role, kind, number, tool_name = _classify(head)

        if role == SYSTEM:
            system_text = body
            turns.append(Turn(role=SYSTEM, number=None, text=body))
            continue

        if kind == "call":
            # Маркерный формат: аргументы лежат телом блока.
            try:
                parsed = json.loads(body.strip())
            except json.JSONDecodeError:
                parsed = None
            call = ToolCall(
                name=tool_name or "",
                args=parsed if isinstance(parsed, dict) else {},
                raw=body.strip(),
                requestor=role,
            )
            all_calls.append(call)
            turns.append(Turn(role=role, number=number, text=body, tool_calls=(call,)))
            continue

        if kind == "result":
            turns.append(
                Turn(role=role, number=number, text=body, tool_responses=((tool_name or "", body),))
            )
            continue

        calls, resps = _parse_inline_calls(body, role)
        all_calls.extend(calls)
        turns.append(
            Turn(role=role, number=number, text=body, tool_calls=calls, tool_responses=resps)
        )

    if strict and not system_text:
        raise TranscriptError("не найден блок ⟦SYSTEM⟧ — без политики и схем проверять нечего")

    tools = parse_tool_schemas(system_text)
    return Transcript(
        system=system_text,
        policy=_extract_block(system_text, "policy"),
        instructions=_extract_block(system_text, "instructions"),
        tools=tools,
        turns=turns,
        health=_health(len(markers), tools, all_calls),
    )


def parse_response(response: str, *, strict: bool = True) -> Turn:
    """Разобрать проверяемый ответ — это ровно один ход ассистента.

    Если маркера нет, разбор идёт по сырому тексту, но у возвращённого хода
    `marker_found = False`. Потребитель обязан это проверять: тихий ноль вызовов при
    сменившемся формате выглядит как «агент ничего не делал» и гасит все проверки.
    """
    _, embedded = strip_document_tags(response)
    if embedded is not None:
        response = embedded

    t = parse_transcript(response, strict=False)
    assistant = [x for x in t.turns if x.role == ASSISTANT]
    if assistant:
        first = assistant[0]
        calls = tuple(c for x in assistant for c in x.tool_calls)
        resps = tuple(r for x in assistant for r in x.tool_responses)
        return Turn(
            role=ASSISTANT,
            number=first.number,
            text=first.text,
            tool_calls=calls,
            tool_responses=resps,
            marker_found=True,
        )

    if strict:
        raise TranscriptError(f"в response не найден ход ассистента: {response[:200]!r}")
    calls, resps = _parse_inline_calls(response, ASSISTANT)
    return Turn(
        role=ASSISTANT,
        number=None,
        text=response,
        tool_calls=calls,
        tool_responses=resps,
        marker_found=False,
    )

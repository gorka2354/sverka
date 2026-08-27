"""Разбор транскрипта.

⚠️ Формат восстановлен по фактическому содержимому `data/raw/valid.parquet` (27.08.2026),
а НЕ по описанию задачи — они расходятся. В описании обещаны маркеры вида
`⟦ASSISTANT_TOOL_CALL name="X"⟧` и `⟦TOOL_RESULT name="X" requestor="assistant"⟧`;
в выданных данных их нет.

Фактическая разметка:

    ⟦SYSTEM⟧
    <instructions> … </instructions>
    <policy> … </policy>
    [AVAILABLE TOOLS]
    - book_reservation — Book a reservation.
        user_id: string! — The ID of the user…
        cabin: string! [enum: business|economy|basic_economy] — …
        flights: array! — An array of objects…
            · flight_number: string! — Flight number.

    ⟦ASSISTANT⟧            ← приветствие, без номера хода
    ⟦USER⟧
    ⟦ASSISTANT · ход 2⟧
    →· TOOL_CALL get_user_details: {"user_id": "sofia_kim_7287"}
    ←· TOOL_RESPONSE get_user_details: {…}

Вызовы инструментов лежат ВНУТРИ тела хода строками с префиксами `→ TOOL_CALL` и
`← TOOL_RESPONSE`, а `!` в схеме означает обязательный параметр.

На реальном `test.csv` формат может отличаться и от того, и от другого — поэтому парсер
падает громко (`TranscriptError`), а не возвращает пустой разбор: молчаливый пустой результат
означал бы съеденный сабмит с мусорными предсказаниями.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

LB, RB = "⟦", "⟧"  # ⟦ ⟧
_MARKER_RE = re.compile(rf"{LB}([^{RB}]{{0,80}}){RB}")
_TURN_NO_RE = re.compile(r"ход\s+(\d+)")
_TOOL_CALL_RE = re.compile(r"[→>]\s*TOOL_CALL\s+([A-Za-z0-9_]+)\s*:\s*(\{.*?\})\s*(?=\n|$)", re.S)
_TOOL_RESP_RE = re.compile(r"[←<]\s*TOOL_RESPONSE\s+([A-Za-z0-9_]+)\s*:\s*(.*?)\s*(?=\n\s*[→←>]|\n\s*\n|$)", re.S)
_TOOLS_HEADER = "[AVAILABLE TOOLS]"


class TranscriptError(RuntimeError):
    """Транскрипт не разобрался. Падаем громко и с текстом, а не молча."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    raw: str

    @property
    def parsed_ok(self) -> bool:
        """JSON аргументов распарсился (пустой объект тоже считается валидным)."""
        return bool(self.args) or self.raw.strip() in ("{}", "")


@dataclass(frozen=True)
class ParamSpec:
    """Один параметр в схеме инструмента из [AVAILABLE TOOLS]."""

    name: str
    type: str
    required: bool
    enum: tuple[str, ...] = ()
    description: str = ""
    depth: int = 0


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


@dataclass(frozen=True)
class Turn:
    role: str  # SYSTEM | USER | ASSISTANT
    number: int | None  # номер хода (у приветствия и реплик пользователя None)
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_responses: tuple[tuple[str, str], ...] = ()


@dataclass
class Transcript:
    system: str
    policy: str
    instructions: str
    tools: dict[str, ToolSchema] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "ASSISTANT"]

    def turns_before(self, number: int) -> list[Turn]:
        """Ходы до указанного номера — история, доступная агенту на проверяемом ходу."""
        out = []
        for t in self.turns:
            if t.number is not None and t.number >= number:
                break
            out.append(t)
        return out


def _extract_block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else ""


def parse_tool_schemas(system_text: str) -> dict[str, ToolSchema]:
    """Разобрать блок [AVAILABLE TOOLS] в словарь схем.

    Формат строки инструмента:  `- name — description`
    Формат параметра:           `    name: type! [enum: a|b] — description`
    Вложенные поля помечены `·` и большей глубиной отступа.
    Восклицательный знак после типа означает обязательность.
    """
    i = system_text.find(_TOOLS_HEADER)
    if i < 0:
        return {}
    body = system_text[i + len(_TOOLS_HEADER):]

    tools: dict[str, ToolSchema] = {}
    cur_name: str | None = None
    cur_desc = ""
    cur_params: list[ParamSpec] = []

    def flush() -> None:
        if cur_name:
            tools[cur_name] = ToolSchema(cur_name, cur_desc, tuple(cur_params))

    for line in body.splitlines():
        if not line.strip():
            continue
        m_tool = re.match(r"^-\s+([A-Za-z0-9_]+)\s*[—-]\s*(.*)$", line.strip())
        if m_tool and not line.startswith((" ", "\t")):
            flush()
            cur_name, cur_desc, cur_params = m_tool.group(1), m_tool.group(2).strip(), []
            continue
        if cur_name is None:
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip().lstrip("·").strip()
        m_par = re.match(
            r"^([A-Za-z0-9_]+)\s*:\s*([A-Za-z_]+)(!?)\s*(?:\[enum:\s*([^\]]+)\])?\s*(?:[—-]\s*(.*))?$",
            stripped,
        )
        if m_par:
            cur_params.append(
                ParamSpec(
                    name=m_par.group(1),
                    type=m_par.group(2),
                    required=m_par.group(3) == "!",
                    enum=tuple(x.strip() for x in m_par.group(4).split("|")) if m_par.group(4) else (),
                    description=(m_par.group(5) or "").strip(),
                    depth=0 if indent <= 4 else 1,
                )
            )
    flush()
    return tools


def _parse_tool_calls(body: str) -> tuple[tuple[ToolCall, ...], tuple[tuple[str, str], ...]]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(body):
        raw = m.group(2)
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(name=m.group(1), args=args if isinstance(args, dict) else {}, raw=raw))
    resps = tuple((m.group(1), m.group(2)) for m in _TOOL_RESP_RE.finditer(body))
    return tuple(calls), resps


def parse_transcript(prompt: str, *, strict: bool = True) -> Transcript:
    """Разобрать поле `prompt` в структуру.

    strict=True (по умолчанию): нет маркеров или нет системного блока → TranscriptError.
    В боевом `predict.py` вызывать со strict=False и логировать, чтобы один битый пример
    не ронял весь прогон.
    """
    markers = list(_MARKER_RE.finditer(prompt))
    if not markers:
        if strict:
            raise TranscriptError(
                "в prompt не найдено ни одного маркера ⟦…⟧ — формат отличается от ожидаемого; "
                f"первые 200 символов: {prompt[:200]!r}"
            )
        return Transcript(system="", policy="", instructions="", turns=[])

    turns: list[Turn] = []
    system_text = ""
    for idx, m in enumerate(markers):
        head = m.group(1).strip()
        start = m.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(prompt)
        body = prompt[start:end].strip("\n")

        role = head.split("·")[0].strip().upper()
        num_m = _TURN_NO_RE.search(head)
        number = int(num_m.group(1)) if num_m else None

        if role == "SYSTEM":
            system_text = body
            turns.append(Turn(role="SYSTEM", number=None, text=body))
            continue

        calls, resps = _parse_tool_calls(body) if role == "ASSISTANT" else ((), ())
        turns.append(Turn(role=role, number=number, text=body, tool_calls=calls, tool_responses=resps))

    if strict and not system_text:
        raise TranscriptError("не найден блок ⟦SYSTEM⟧ — без политики и схем проверять нечего")

    return Transcript(
        system=system_text,
        policy=_extract_block(system_text, "policy"),
        instructions=_extract_block(system_text, "instructions"),
        tools=parse_tool_schemas(system_text),
        turns=turns,
    )


def parse_response(response: str, *, strict: bool = True) -> Turn:
    """Разобрать проверяемый ответ — это ровно один ход ассистента."""
    t = parse_transcript(response, strict=False)
    assistant = [x for x in t.turns if x.role == "ASSISTANT"]
    if not assistant:
        if strict:
            raise TranscriptError(f"в response не найден ход ассистента: {response[:200]!r}")
        calls, resps = _parse_tool_calls(response)
        return Turn(role="ASSISTANT", number=None, text=response, tool_calls=calls, tool_responses=resps)
    return assistant[0]

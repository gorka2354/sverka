"""Проверка аргументов вызова против схемы инструмента из `[AVAILABLE TOOLS]`.

Зачем рекурсивно. `[замер]` Плоская проверка (только поля верхнего уровня) на выданном
сэмпле не сработала **ни разу**: единственное схемное нарушение — `airline__23::t10`, где
в `payment_methods` переданы поля `id` и `source` вместо обязательных `payment_id`
и `amount`, — лежит внутри объекта внутри массива. Категория «неверные аргументы», которую
ТЗ называет явно, плоским валидатором не ловится вовсе.

Почему пришлось менять и структуру схемы, а не только проверку. Разбор
`[AVAILABLE TOOLS]` возвращает **плоский** кортеж `ParamSpec` с полем `depth ∈ {0,1}`.
Без ссылки на родителя `payment_id` внутри `payment_methods` неотличим от `first_name`
внутри `passengers` — у обоих `depth = 1`, и валидатор проверял бы поля не того объекта.
Поэтому `ParamSpec.parent` заполняется при разборе, а здесь используется.

Осторожность важнее полноты. Ложное срабатывание тут стоит дороже пропуска: список того,
что ТЗ галлюцинацией НЕ считает, — это ровно перечень ложных тревог наивного валидатора.
Поэтому проверка молчит везде, где схема недостаточно определена: нет разобранных полей
уровня, тип не распознан, значение `null`. Проверяется только то, что схема утверждает
явно.
"""

from __future__ import annotations

from dataclasses import dataclass

from sverka.transcript import ParamSpec, ToolCall, ToolSchema

# Соответствие типов схемы питоновским. Намеренно снисходительное: `integer` принимает
# `bool`? нет — bool исключён явно, потому что True вместо 1 это настоящая ошибка;
# а вот `number` принимает и int, и float, потому что различать их бессмысленно.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "str": (str,),
    "integer": (int,),
    "int": (int,),
    "number": (int, float),
    "float": (int, float),
    "boolean": (bool,),
    "bool": (bool,),
    "array": (list, tuple),
    "list": (list, tuple),
    "object": (dict,),
    "dict": (dict,),
}


# Сила улики. Разделение сделано по замеру, а не по вкусу.
#
# `[замер]` На выданном сэмпле только `missing_required` срабатывает ровно на том
# единственном примере (`airline__23::t10`), объяснение которого прямо ссылается на схему:
# «обязательное поле amount вообще отсутствует, а имена id/source в схеме не предусмотрены».
# `unknown_field` срабатывает на трёх, но у двух из них организаторы назвали совсем другие
# причины (расчёт багажа, правила отмены), а лишними были описательные поля `origin`,
# `destination`, `price` внутри `flights[]` при всех обязательных на месте. Это похоже на
# верный ответ по неверной причине — ровно та ошибка, за которую уже пришлось выбросить
# гейт «два любых вызова».
#
# Поэтому: пропущенное обязательное поле — сильная улика; лишнее поле при полном комплекте
# обязательных — подсказка судье, не более.
STRONG_KINDS = ("missing_required", "wrong_type", "bad_enum")
WEAK_KINDS = ("unknown_field",)


@dataclass(frozen=True)
class SchemaViolation:
    kind: str  # missing_required | unknown_field | wrong_type | bad_enum
    path: str
    detail: str

    @property
    def strong(self) -> bool:
        return self.kind in STRONG_KINDS

    def __str__(self) -> str:
        mark = "сильная" if self.strong else "слабая"
        return f"{self.kind} ({mark}) в {self.path}: {self.detail}"


def _type_ok(value, declared: str) -> bool | None:
    """True/False, либо None если тип схемы нам неизвестен и судить не о чем."""
    expected = _TYPE_MAP.get(declared.lower())
    if expected is None:
        return None
    if isinstance(value, bool) and int in expected and bool not in expected:
        return False  # True вместо 1 — настоящая ошибка, а не поблажка
    return isinstance(value, expected)


def _check_level(
    value: dict,
    specs: list[ParamSpec],
    schema: ToolSchema,
    path: str,
    out: list[SchemaViolation],
) -> None:
    known = {p.name for p in specs}
    if not known:
        return  # уровень схемой не описан — молчим, а не выдумываем нарушения

    for p in specs:
        if p.required and p.name not in value:
            out.append(
                SchemaViolation(
                    "missing_required", f"{path}.{p.name}".lstrip("."),
                    f"обязательное поле отсутствует (тип {p.type})",
                )
            )

    for key, val in value.items():
        full = f"{path}.{key}".lstrip(".")
        if key not in known:
            out.append(
                SchemaViolation(
                    "unknown_field", full,
                    f"поле не предусмотрено схемой; схема ожидает {sorted(known)}",
                )
            )
            continue
        spec = next(p for p in specs if p.name == key)
        if val is None:
            continue

        ok = _type_ok(val, spec.type)
        if ok is False:
            out.append(
                SchemaViolation(
                    "wrong_type", full,
                    f"объявлен {spec.type}, передан {type(val).__name__}",
                )
            )
            continue

        if spec.enum and isinstance(val, str) and val not in spec.enum:
            out.append(
                SchemaViolation("bad_enum", full, f"значение {val!r} вне {list(spec.enum)}")
            )

        children = [c for c in schema.params if c.parent == spec.name and c.depth > 0]
        if not children:
            continue
        if isinstance(val, dict):
            _check_level(val, children, schema, full, out)
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    _check_level(item, children, schema, f"{full}[{i}]", out)


def validate_call(call: ToolCall, schema: ToolSchema) -> list[SchemaViolation]:
    """Проверить один вызов против его схемы.

    Возвращает пустой список, если проверять нечего: аргументы не разобрались (тогда это
    забота другого детектора, а не наша) или схема не содержит полей верхнего уровня.
    """
    if not call.parsed_ok:
        return []
    top = [p for p in schema.params if p.depth == 0]
    if not top:
        return []
    out: list[SchemaViolation] = []
    _check_level(call.args, top, schema, "", out)
    return out


def validate_turn(turn, tools: dict[str, ToolSchema]) -> list[tuple[str, SchemaViolation]]:
    """Проверить все вызовы хода. Вызовы неизвестных инструментов пропускаются —
    это зона гейта «недоступный инструмент», и дублировать его вердикт незачем."""
    out: list[tuple[str, SchemaViolation]] = []
    for c in turn.tool_calls:
        schema = tools.get(c.name)
        if schema is None:
            continue
        for v in validate_call(c, schema):
            out.append((c.name, v))
    return out

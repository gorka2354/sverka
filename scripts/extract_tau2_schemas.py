"""Извлечь схемы инструментов из исходников tau2-bench без установки пакета.

Зачем. В публичных траекториях поле `tool_defs` пусто, а у нас из репозитория взяты только
имена инструментов. Без параметров рекурсивный валидатор аргументов на добытых данных
не работает вовсе, и блок `[AVAILABLE TOOLS]` в собранном корпусе получается беднее, чем
у организаторов.

Почему разбором исходника, а не импортом. Установка tau2-bench тянет тяжёлые зависимости
ради того, чтобы прочитать сигнатуры функций. Разбор через `ast` даёт то же самое: имя,
аннотацию типа, наличие значения по умолчанию (то есть обязательность) и описание
из docstring — и не требует ни одной сторонней библиотеки.

Границы честно. Из аннотаций восстанавливается тип и обязательность; вложенность
восстанавливается только для типов вида `list[Something]`, где `Something` — класс модели
данных того же домена, и только на один уровень. Значения `enum` берутся из `Literal[...]`
и из классов-перечислений, если они лежат рядом. Всё, что не восстановилось, помечается
пустым, а не угадывается.

    python scripts/extract_tau2_schemas.py --repo ~/sverka/tau2-bench --out schemas.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")

# Соответствие питоновских аннотаций типам, которыми организаторы печатают схему.
_TYPE_WORDS = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "List": "array",
    "Dict": "object",
}


def _first_model_name(node: ast.AST | None) -> str | None:
    """Имя класса модели внутри `List[X]` или `List[X | dict]`.

    Второй случай — рабочий: в tau2-bench аргументы объявлены как `List[Passenger | dict]`,
    то есть внутри подписки лежит `BinOp`, а не простое имя. Наивный разбор находил там
    ничего и терял всю вложенность, из-за которой рекурсивная проверка и затевалась.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp):
        for side in (node.left, node.right):
            n = _first_model_name(side)
            if n and n != "dict":
                return n
    return None


def _is_required(default: ast.AST | None) -> bool:
    """Обязателен ли параметр.

    Тонкость Pydantic: `payment_id: str = Field(description="...")` выглядит как параметр
    со значением по умолчанию, но обязателен — умолчания нет, есть только метаданные.
    Обратное верно для `Field(default=..., ...)` и `Field(default_factory=...)`.
    """
    if default is None:
        return True
    if isinstance(default, ast.Call):
        fn = getattr(default.func, "id", "") or getattr(default.func, "attr", "")
        if fn == "Field":
            return not any(k.arg in ("default", "default_factory") for k in default.keywords)
    return False


def _ann_to_type(node: ast.AST | None) -> tuple[str, str | None]:
    """Аннотация → (тип для схемы, имя вложенного класса, если это список объектов)."""
    if node is None:
        return "", None
    if isinstance(node, ast.Name):
        return _TYPE_WORDS.get(node.id, node.id), None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _TYPE_WORDS.get(node.value, node.value), None
    if isinstance(node, ast.Subscript):
        base, _ = _ann_to_type(node.value)
        inner = node.slice
        # Optional[X] / Union[X, None] — разворачиваем к X, обязательность решает default
        if base in ("Optional", "Union"):
            parts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
            for p in parts:
                t, nested = _ann_to_type(p)
                if t and t != "None":
                    return t, nested
            return "", None
        if base == "Literal":
            return "string", None
        if base in ("array", "list"):
            return "array", _first_model_name(inner)
        return base, None
    if isinstance(node, ast.Attribute):
        return _TYPE_WORDS.get(node.attr, node.attr), None
    return "", None


def _literal_values(node: ast.AST | None) -> tuple[str, ...]:
    if isinstance(node, ast.Subscript):
        base = getattr(node.value, "id", "") or getattr(node.value, "attr", "")
        if base == "Literal":
            sl = node.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            return tuple(
                str(e.value) for e in elts if isinstance(e, ast.Constant) and e.value is not None
            )
    return ()


def _param_docs(doc: str | None) -> dict[str, str]:
    """Описания параметров из docstring в стиле Google/Args."""
    if not doc:
        return {}
    out: dict[str, str] = {}
    m = re.search(r"\n\s*Args:\s*\n(.*?)(?:\n\s*(?:Returns|Raises|Yields):|\Z)", doc, re.S)
    if not m:
        return out
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*([A-Za-z0-9_]+)\s*(?:\([^)]*\))?\s*:\s*(.*)", line)
        if mm:
            out[mm.group(1)] = mm.group(2).strip()
    return out


def _class_fields(tree: ast.Module) -> dict[str, list[dict]]:
    """Поля классов модели данных — нужны, чтобы раскрыть `list[Passenger]` на один уровень."""
    out: dict[str, list[dict]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        fields = []
        for st in node.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                t, _ = _ann_to_type(st.annotation)
                fields.append(
                    {
                        "name": st.target.id,
                        "type": t,
                        "required": _is_required(st.value),
                        "enum": list(_literal_values(st.annotation)),
                    }
                )
        if fields:
            out[node.name] = fields
    return out


def extract_domain(repo: Path, domain: str) -> dict:
    dom_dir = repo / "src" / "tau2" / "domains" / domain
    models: dict[str, list[dict]] = {}
    for f in sorted(dom_dir.glob("*.py")):
        try:
            models.update(_class_fields(ast.parse(f.read_text(encoding="utf-8"))))
        except SyntaxError:
            continue

    tools: dict[str, dict] = {}
    for f in sorted(dom_dir.glob("*.py")):
        side = "user" if f.name == "user_tools.py" else "agent"
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(
                (isinstance(d, ast.Call) and getattr(d.func, "id", "") == "is_tool")
                or getattr(d, "id", "") == "is_tool"
                for d in node.decorator_list
            ):
                continue
            doc = ast.get_docstring(node) or ""
            pdocs = _param_docs(doc)
            args = node.args
            defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
            params = []
            for a, d in zip(args.args, defaults):
                if a.arg == "self":
                    continue
                t, nested = _ann_to_type(a.annotation)
                p = {
                    "name": a.arg,
                    "type": t,
                    "required": _is_required(d),
                    "enum": list(_literal_values(a.annotation)),
                    "description": pdocs.get(a.arg, ""),
                    "children": [],
                }
                if nested and nested in models:
                    p["children"] = models[nested]
                params.append(p)
            tools[node.name] = {
                "side": side,
                "description": doc.split("\n")[0].strip(),
                "params": params,
            }
    return tools


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).expanduser()
    out: dict = {"_meta": {"repo": "sierra-research/tau2-bench", "license": "MIT"}}
    for dom in DOMAINS:
        tools = extract_domain(repo, dom)
        out[dom] = tools
        agent = [n for n, t in tools.items() if t["side"] == "agent"]
        user = [n for n, t in tools.items() if t["side"] == "user"]
        with_params = sum(1 for t in tools.values() if t["params"])
        nested = sum(1 for t in tools.values() for p in t["params"] if p["children"])
        print(
            f"{dom:20s} агент={len(agent):2d} пользователь={len(user):2d} "
            f"со схемой={with_params:2d} с вложенностью={nested}"
        )

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записано в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Локальный стенд: прогнать детектор по размеченным данным и напечатать метрики.

Все решения принимаются по его выводу, а сабмит тратится только на то, что здесь проверить
нельзя.

**Метрики печатаются по стратам, и это не украшение.** `[замер]` Правило «в ответе есть хотя
бы один TOOL_CALL → метка 1» даёт на выданном сэмпле F1 = 0.913: из 23 ответов без вызовов
21 отрицательный, из 23 ответов с вызовами 21 положительный. Почти наверняка это артефакт
нарезки сэмпла. Пока он не опровергнут сабмитом, **общая метрика на этих 46 примерах меряет
умение угадать наличие вызова, а не наличие галлюцинации**, и сравнивать варианты по ней
нельзя. Сравнивать нужно внутри страт «с вызовом» / «без вызова», по 23 примера в каждой.

Две контрольные строки печатаются всегда: артефакт TOOL_CALL и счётчик ходов (F1 = 0.724).
Обе — планки, а не кандидаты. Признак «номер хода» использовать в детекторе запрещено:
пять отрицательных примеров сидят на ходу t1, это свойство набора негативов, а не задачи.

    python scripts/evaluate.py --detector gates
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.data import Example, load_examples  # noqa: E402
from sverka.gates import apply_gates, load_known_user_tools  # noqa: E402
from sverka.metrics import bootstrap_f1_grouped, score  # noqa: E402
from sverka.transcript import parse_response, parse_transcript  # noqa: E402

_USER_TOOLS = load_known_user_tools()


def _parsed(ex: Example):
    return parse_transcript(ex.prompt, strict=False), parse_response(ex.response, strict=False)


DETECTORS = {
    "constant1": lambda ex, tr, tn: 1,
    "constant0": lambda ex, tr, tn: 0,
    "has_tool_call": lambda ex, tr, tn: 1 if tn.tool_calls else 0,
    "turn_ge5": lambda ex, tr, tn: 1 if (ex.turn or 0) >= 5 else 0,
    "gates": lambda ex, tr, tn: 1
    if apply_gates(tr, tn, user_tools=_USER_TOOLS.get(ex.domain, set())).fired
    else 0,
}

CONTROLS = ("has_tool_call", "turn_ge5")


def report(name: str, examples: list[Example], preds: list[int], gold: list[int]) -> None:
    groups = [e.conv for e in examples]
    s = score(gold, preds)
    lo, hi = bootstrap_f1_grouped(gold, preds, groups)
    print(f"детектор: {name}")
    print(f"  ОБЩЕЕ (не принимать как аргумент, см. страты): {s}")
    print(f"    F1 95% интервал по диалогам: [{lo:.3f}, {hi:.3f}]  ширина {hi-lo:.3f}")

    strata = {
        "ответ С вызовом": [i for i, e in enumerate(examples) if _parsed(e)[1].tool_calls],
        "ответ БЕЗ вызова": [i for i, e in enumerate(examples) if not _parsed(e)[1].tool_calls],
    }
    print("  ПО СТРАТАМ — здесь и надо сравнивать варианты:")
    for label, idx in strata.items():
        if not idx:
            continue
        st = score([gold[i] for i in idx], [preds[i] for i in idx])
        base = sum(gold[i] for i in idx) / len(idx)
        print(f"    {label:18s} n={len(idx):2d} доля позитива={base:.2f}  {st}")

    domains = sorted({e.domain for e in examples})
    if len(domains) > 1:
        print("  по доменам:")
        for d in domains:
            idx = [i for i, e in enumerate(examples) if e.domain == d]
            sd = score([gold[i] for i in idx], [preds[i] for i in idx])
            print(f"    {d:20s} {sd}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/valid.parquet")
    ap.add_argument("--detector", default="gates", help=f"один из {list(DETECTORS)} или 'all'")
    args = ap.parse_args()

    examples = [e for e in load_examples(args.data) if e.label is not None]
    if not examples:
        raise SystemExit("в данных нет колонки label — оценивать нечего")
    gold = [int(e.label) for e in examples]  # type: ignore[arg-type]
    print(
        f"загружено: {len(examples)} примеров, {len({e.conv for e in examples})} диалогов, "
        f"доля позитива {sum(gold)/len(gold):.2f}",
        file=sys.stderr,
    )

    names = list(DETECTORS) if args.detector == "all" else [args.detector]
    for name in names:
        if name not in DETECTORS:
            raise SystemExit(f"неизвестный детектор {name}; есть: {list(DETECTORS)}")
        fn = DETECTORS[name]
        t0 = time.perf_counter()
        preds = []
        for e in examples:
            tr, tn = _parsed(e)
            preds.append(int(fn(e, tr, tn)))
        dt = time.perf_counter() - t0
        print()
        if name in CONTROLS:
            print("### КОНТРОЛЬ, не кандидат в решение ###")
        report(name, examples, preds, gold)
        print(f"  время: {dt*1000:.0f} мс на {len(examples)} примеров ({dt/len(examples)*1000:.2f} мс/пример)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

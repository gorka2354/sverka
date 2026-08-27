"""Локальный стенд: прогнать детектор по размеченным данным и напечатать метрики.

Это основной инструмент проекта. Все решения принимаются по его выводу, а сабмит тратится
только на то, что здесь проверить нельзя (см. `docs/strategy.md`, §7).

Печатает F1 с бутстрэп-интервалом по диалогам — интервал показывает, значима ли разница
между двумя вариантами, или мы гоняемся за шумом на выборке в 46 примеров.

    python scripts/evaluate.py --data data/raw/valid.parquet --detector constant
"""

from __future__ import annotations

import argparse
import sys
import time

from poverka.data import Example, load_examples
from poverka.detectors import ConstantDetector, RandomDetector, ResponseLengthDetector
from poverka.metrics import bootstrap_f1_grouped, score

DETECTORS = {
    "constant1": lambda: ConstantDetector(value=1),
    "constant0": lambda: ConstantDetector(value=0),
    "random": lambda: RandomDetector(p=0.5),
    "length": lambda: ResponseLengthDetector(),
}


def evaluate(detector, examples: list[Example]) -> None:
    labelled = [e for e in examples if e.label is not None]
    if not labelled:
        raise SystemExit("в данных нет колонки label — оценивать нечего")

    t0 = time.perf_counter()
    preds = [detector.predict(e).label for e in labelled]
    dt = time.perf_counter() - t0

    y_true = [int(e.label) for e in labelled]  # type: ignore[arg-type]
    groups = [e.conv for e in labelled]

    s = score(y_true, preds)
    lo, hi = bootstrap_f1_grouped(y_true, preds, groups)

    print(f"детектор: {getattr(detector, 'name', type(detector).__name__)}")
    print(f"  {s}")
    print(f"  F1 95% интервал (бутстрэп по диалогам): [{lo:.3f}, {hi:.3f}]  ширина {hi-lo:.3f}")
    print(f"  время: {dt*1000:.0f} мс на {len(labelled)} примеров ({dt/len(labelled)*1000:.1f} мс/пример)")

    domains = sorted({e.domain for e in labelled})
    if len(domains) > 1:
        print("  по доменам:")
        for d in domains:
            idx = [i for i, e in enumerate(labelled) if e.domain == d]
            sd = score([y_true[i] for i in idx], [preds[i] for i in idx])
            print(f"    {d:20s} {sd}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/valid.parquet")
    ap.add_argument("--detector", default="all", help=f"один из {list(DETECTORS)} или 'all'")
    args = ap.parse_args()

    examples = load_examples(args.data)
    print(f"загружено: {len(examples)} примеров, {len({e.conv for e in examples})} диалогов", file=sys.stderr)

    names = list(DETECTORS) if args.detector == "all" else [args.detector]
    for name in names:
        if name not in DETECTORS:
            raise SystemExit(f"неизвестный детектор {name}; есть: {list(DETECTORS)}")
        print()
        evaluate(DETECTORS[name](), examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

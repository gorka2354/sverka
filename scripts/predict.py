"""Точка входа для платформы.

Платформа выполняет ровно это:

    pip install .
    python scripts/predict.py --input <path_to_test.csv> --output <path_to_predictions.csv>

Выход — CSV с колонками `id,label`.

Инварианты, которые здесь нельзя нарушать:

1. **Скрипт не имеет права упасть.** Упавший прогон — это не «попробуем ещё раз»: сабмитов
   десять, а время дорого. Любая ошибка на конкретном примере ловится, пишется в stderr,
   примеру ставится безопасное предсказание.
2. **Время печатается всегда.** Лимит платформы — 30 минут на весь набор. Замер должен быть
   виден в логе прогона, а не выясняться постфактум.
3. **Формат входа не гарантирован.** Сэмпл размечен не так, как обещает документация;
   на реальном тесте возможен третий вариант. Парсер вызывается в нестрогом режиме,
   а число неразобранных примеров печатается отдельной строкой — это диагностика,
   по которой видно, что решение работало вслепую.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from sverka.data import load_examples
from sverka.detectors import ConstantDetector

# Метка, которую ставим примеру, если детектор упал на нём.
# 1, а не 0: при балансе 50/50 ложноположительное стоит F1 дешевле, чем пропуск.
FALLBACK_LABEL = 1


def build_detector(name: str):
    """Собрать детектор по имени. Точка расширения для каскада (inc-4)."""
    if name == "constant":
        return ConstantDetector(value=1)
    raise SystemExit(f"неизвестный детектор: {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Guardian of Truth — inference")
    ap.add_argument("--input", required=True, help="путь к test.csv (или .parquet)")
    ap.add_argument("--output", required=True, help="куда писать predictions.csv")
    ap.add_argument("--detector", default="constant", help="какой детектор использовать")
    args = ap.parse_args()

    t0 = time.perf_counter()
    examples = load_examples(args.input)
    print(f"[sverka] загружено примеров: {len(examples)}", file=sys.stderr)

    detector = build_detector(args.detector)

    rows: list[dict] = []
    failures = 0
    for ex in examples:
        try:
            verdict = detector.predict(ex)
            label = int(verdict.label)
        except Exception as e:  # noqa: BLE001 — падать нельзя, см. инвариант 1
            failures += 1
            label = FALLBACK_LABEL
            print(f"[sverka] ОШИБКА на {ex.id}: {type(e).__name__}: {e}", file=sys.stderr)
        rows.append({"id": ex.id, "label": label})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    dt = time.perf_counter() - t0
    per = dt / len(examples) if examples else 0.0
    print(
        f"[sverka] готово за {dt:.1f} c ({per*1000:.0f} мс/пример), "
        f"неразобранных: {failures}, записано: {len(rows)} → {out_path}",
        file=sys.stderr,
    )
    if failures:
        print(
            f"[sverka] ВНИМАНИЕ: {failures} примеров получили запасную метку {FALLBACK_LABEL}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

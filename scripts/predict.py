"""Точка входа для платформы.

Платформа выполняет ровно это:

    pip install .
    python scripts/predict.py --input <path_to_test.csv> --output <path_to_predictions.csv>

Выход — CSV с колонками `id,label`.

Инварианты, которые здесь нельзя нарушать:

1. **Скрипт не имеет права упасть и не имеет права выйти с неполным файлом.** Отработавший
   прогон тратит попытку даже с мусорным результатом, а попыток десять. Любая ошибка на
   конкретном примере ловится, пишется в stderr, примеру ставится запасная метка. При
   исчерпании бюджета времени остаток дописывается запасной меткой, а не обрывается.

2. **Бюджет времени сторожится, а не измеряется постфактум.** Лимит платформы 30 минут.
   Watchdog со ступенчатой деградацией встроен сейчас, пока слой считает 0.5 мс на пример,
   именно потому что позже, с судьёй на локальных весах, переделывать будет некогда.

3. **Молчаливый пустой разбор — отдельно считаемая аварийная ситуация.** Формат сэмпла
   расходится с документацией, документировано три несовместимых варианта. Если разбор
   не удался, детерминированные проверки обязаны молчать (иначе вместо тишины будет поток
   ложных срабатываний), а доля таких примеров — попасть в лог.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from sverka.data import load_examples
from sverka.gates import apply_gates, load_known_user_tools
from sverka.transcript import parse_response, parse_transcript

# Метка, которую ставим примеру, если детектор упал или кончилось время.
# 1, а не 0: при неизвестном балансе классов пропуск стоит F1 дороже ложной тревоги.
FALLBACK_LABEL = 1

# Лимит платформы 30 минут. Сторожим 24, чтобы осталось на запись файла и на хвост.
DEFAULT_BUDGET_SECONDS = 24 * 60


def detector_constant(value: int):
    def run(ex, tr, turn) -> int:
        return value

    return run


def detector_has_tool_call(ex, tr, turn) -> int:
    """Контрольная планка: «в ответе есть вызов инструмента → галлюцинация».

    На выданном сэмпле даёт F1 = 0.913 — почти наверняка артефакт того, как нарезался
    сэмпл (позитивы из ходов-действий, негативы из текстовых). Существует как эталон
    для сравнения и как содержимое отдельного сабмита, проверяющего перенос артефакта.
    """
    return 1 if turn.tool_calls else 0


def detector_gates(ex, tr, turn) -> int:
    """Детерминированные гейты.

    ⚠️ Точность гейтов на выданном сэмпле не измерена: достижимых отрицательных в страте
    «ответ с вызовом» ноль. Этот детектор существует для замеров и для сабмита, который
    эту точность и померит; в финальном решении гейты идут признаком для судьи, а не
    самостоятельной меткой.
    """
    rep = apply_gates(tr, turn, user_tools=_USER_TOOLS.get(ex.domain, set()))
    return 1 if rep.fired else 0


DETECTORS = {
    "constant1": detector_constant(1),
    "constant0": detector_constant(0),
    "has_tool_call": detector_has_tool_call,
    "gates": detector_gates,
}

_USER_TOOLS = load_known_user_tools()


def write_predictions(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Guardian of Truth — inference")
    ap.add_argument("--input", required=True, help="путь к test.csv (или .parquet)")
    ap.add_argument("--output", required=True, help="куда писать predictions.csv")
    ap.add_argument("--detector", default="gates", help=f"один из {list(DETECTORS)}")
    ap.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    args = ap.parse_args()

    if args.detector not in DETECTORS:
        print(f"[sverka] неизвестный детектор {args.detector!r}; есть: {list(DETECTORS)}", file=sys.stderr)
        return 2
    run = DETECTORS[args.detector]

    t0 = time.perf_counter()
    examples = load_examples(args.input)
    out_path = Path(args.output)
    print(f"[sverka] загружено примеров: {len(examples)}, детектор: {args.detector}", file=sys.stderr)

    rows: list[tuple[str, int]] = []
    failures = 0
    unhealthy = 0
    no_marker = 0
    budget_hit = False

    for i, ex in enumerate(examples):
        if time.perf_counter() - t0 > args.budget_seconds:
            # Бюджет исчерпан: остаток дописывается запасной меткой. Выйти с неполным
            # файлом нельзя — это гарантированно нулевой результат за потраченную попытку.
            budget_hit = True
            rows.extend((e.id, FALLBACK_LABEL) for e in examples[i:])
            print(
                f"[sverka] ВНИМАНИЕ: бюджет {args.budget_seconds:.0f} c исчерпан на примере {i}; "
                f"оставшимся {len(examples) - i} проставлена запасная метка {FALLBACK_LABEL}",
                file=sys.stderr,
            )
            break
        try:
            tr = parse_transcript(ex.prompt, strict=False)
            turn = parse_response(ex.response, strict=False)
            if tr.health is not None and not tr.health.ok:
                unhealthy += 1
            if not turn.marker_found:
                no_marker += 1
            label = int(run(ex, tr, turn))
        except Exception as e:  # noqa: BLE001 — падать нельзя, см. инвариант 1
            failures += 1
            label = FALLBACK_LABEL
            print(f"[sverka] ОШИБКА на {ex.id}: {type(e).__name__}: {e}", file=sys.stderr)
        rows.append((ex.id, label))

    write_predictions(out_path, rows)

    dt = time.perf_counter() - t0
    per = dt / len(examples) if examples else 0.0
    positives = sum(1 for _, v in rows if v == 1)
    print(
        f"[sverka] готово за {dt:.1f} c ({per*1000:.1f} мс/пример), записано {len(rows)} → {out_path}",
        file=sys.stderr,
    )
    print(
        f"[sverka] предсказано единиц: {positives}/{len(rows)} ({100*positives/max(1,len(rows)):.1f}%)",
        file=sys.stderr,
    )
    if unhealthy or no_marker or failures or budget_hit:
        print(
            f"[sverka] ДИАГНОСТИКА: нездоровых разборов {unhealthy}, ответов без маркера {no_marker}, "
            f"исключений {failures}, бюджет исчерпан: {budget_hit}",
            file=sys.stderr,
        )
    if unhealthy > len(examples) * 0.2:
        print(
            "[sverka] ВНИМАНИЕ: больше пятой части примеров разобрано нездорово — "
            "вероятно, формат теста отличается от ожидаемого, детерминированный слой молчал",
            file=sys.stderr,
        )
    if len(rows) != len(examples):
        print(
            f"[sverka] КРИТИЧНО: строк {len(rows)} против примеров {len(examples)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

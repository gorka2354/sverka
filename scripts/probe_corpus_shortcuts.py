"""Приёмочная проба корпуса на короткие пути. Блокирующая: без неё корпус не принимается.

Зачем. Выданный сэмпл уже один раз обманул нас: правило «в ответе есть TOOL_CALL» даёт там
F1 = 0.913, то есть общая метрика мерила наличие вызова, а не наличие галлюцинации. Если
собственный корпус несёт свой артефакт, мы просто заменим один самообман другим — и не
заметим, потому что цифры вырастут.

Проба ищет признаки, по которым позитивы отделяются от негативов БЕЗ понимания задачи:
наличие вызова инструмента, домен, длина ответа, номер хода, модель-источник. Для каждого
считается доля позитива в группах. Резкий разброс = короткий путь.

## Что показал первый прогон корпуса v2 — и почему он не принят

`[замер 28.08.2026, 316 524 строки, из них 181 546 размеченных]`

    признак                группы                            доля позитива
    есть вызов инструмента  да / нет                          0.0527 / 0.0000
    домен                   banking / telecom / airline / retail
                                                  0.1140 / 0.0221 / 0.0004 / 0.0003
    номер хода              t>=15 / t5-15 / t<5               0.0526 / 0.0092 / 0.0017
    модель-источник         claude-3-7-sonnet / claude-opus-5 0.0103 / 0.0000

Четыре коротких пути, и первые два убийственные.

**Ход без вызова инструмента — гарантированный негатив.** Из 4 924 позитивов ровно ноль
лежит в текстовой страте. Это не свойство задачи, а следствие конструкции: все позитивы
добыты гейтами, а каждый гейт требует вызова. Детектор, обученный или померенный на таком
корпусе, выучит «нет вызова — нет галлюцинации» и провалит ровно ту половину задачи,
ради которой корпус и строился.

**Домен решает почти всё.** В banking доля позитива в 300 раз выше, чем в retail
(0.114 против 0.0003). Из 4 924 позитивов 3 237 в banking и 1 662 в telecom; на airline
приходится 9, на retail 16. Причина та же: гейт «недоступный инструмент» срабатывает
в telecom (пользовательская диагностика) и в banking (discoverable-инструменты), а в
airline и retail ему нечего ловить.

**Вывод: позитивы этого корпуса циркулярны.** Они выведены гейтами, поэтому на них нельзя
ни мерить гейты, ни учить детектор — он воспроизведёт гейты и доменный перекос.
Негативы при этом остаются ценными: 176 622 надёжных отрицательных примера по обеим
стратам закрывают ту половину измерения, которая на выданном сэмпле не определена вовсе.

## Что с этим делать

Позитивы добывать судьёй из пула кандидатов (134 978 строк), а не гейтами. Негативы
подбирать СОГЛАСОВАННО с позитивами по домену, наличию вызова и позиции хода — иначе
согласование окажется мнимым и разброс вернётся.

    python scripts/probe_corpus_shortcuts.py --index corpus_v2.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Порог, при котором признак считается коротким путём: отношение максимальной доли позитива
# к минимальной по группам. Тройка выбрана так, чтобы естественная неоднородность доменов
# (она есть и в целевых данных) не срабатывала, а перекос на порядок — срабатывал.
SHORTCUT_RATIO = 3.0
MIN_GROUP = 50


def bucket_turn(v: str) -> str:
    n = int(v or 0)
    return "t<5" if n < 5 else ("t5-15" if n < 15 else "t>=15")


def bucket_len(v: str) -> str:
    n = int(v or 0)
    return "<300" if n < 300 else ("300-1000" if n < 1000 else ">=1000")


FEATURES = {
    "есть вызов инструмента": lambda r: "да" if r["has_tool_calls"] == "1" else "нет",
    "домен": lambda r: r["domain"],
    "длина ответа": lambda r: bucket_len(r["response_chars"]),
    "номер хода": lambda r: bucket_turn(r["turn_idx"]),
    "модель-источник": lambda r: r["source_file"].split("_")[0][:22],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--max-groups", type=int, default=8)
    args = ap.parse_args()

    csv.field_size_limit(2**31 - 1)
    with open(args.index, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] in ("0", "1")]
    pos = sum(1 for r in rows if r["label"] == "1")
    print(f"размеченных строк: {len(rows)}, позитивов {pos} (доля {pos/max(1,len(rows)):.4f})")

    verdicts: list[tuple[str, float, str]] = []
    for name, fn in FEATURES.items():
        c: collections.Counter = collections.Counter((fn(r), r["label"]) for r in rows)
        groups = sorted({k[0] for k in c})
        rates = []
        print(f"\n{name}:")
        for g in groups[: args.max_groups]:
            p, n = c[(g, "1")], c[(g, "0")]
            if p + n < MIN_GROUP:
                continue
            rate = p / (p + n)
            rates.append((g, rate))
            print(f"  {g:24s} позитивов {p:6d}, негативов {n:7d}, доля {rate:.4f}")
        if len(rates) < 2:
            continue
        hi = max(rates, key=lambda x: x[1])
        lo = min(rates, key=lambda x: x[1])
        if lo[1] == 0 and hi[1] > 0:
            verdicts.append((name, float("inf"), f"группа {lo[0]!r} НЕ содержит позитивов вовсе"))
        else:
            ratio = hi[1] / lo[1] if lo[1] else float("inf")
            if ratio >= SHORTCUT_RATIO:
                verdicts.append((name, ratio, f"{hi[0]!r} против {lo[0]!r}"))

    print("\n" + "=" * 72)
    if not verdicts:
        print("КОРОТКИХ ПУТЕЙ НЕ НАЙДЕНО — корпус проходит приёмку.")
        return 0
    print("КОРПУС НЕ ПРИНЯТ. Найдены короткие пути:")
    for name, ratio, detail in sorted(verdicts, key=lambda x: -x[1]):
        r = "∞" if ratio == float("inf") else f"×{ratio:.0f}"
        print(f"  {name:24s} перекос {r:>6s}  ({detail})")
    print(
        "\nПризнак, по которому позитивы отделяются без понимания задачи, детектор выучит\n"
        "вместо самой задачи. Прежде чем принимать корпус: добывать позитивы не гейтами,\n"
        "а судьёй по пулу кандидатов, и подбирать негативы согласованно по этим признакам."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

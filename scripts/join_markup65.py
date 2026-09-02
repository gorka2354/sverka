"""Приклеить настоящие метки к уже сделанным прогонам судьи на 65 размеченных ходах.

Зачем. Прогоны `judge_nl_candidates` и `judge_nl_clean` делались ДО разметки, и метка
в них проставлена происхождением строки — «кандидат» значило 1, «чистая» значило 0.
После трёх машинных проходов `[§0е]` метки известны по-настоящему: из 50 кандидатов
положительных оказался 31, остальные 19 — ходы, где требование провалилось не здесь.
Отчёт по старому файлу считает поэтому не то, что нужно.

Скрипт ничего не спрашивает у модели: он берёт готовые вердикты и меняет только метку.
Это даёт точку сравнения «9B на тех же 65 строках при том же окне», без которой прогон
модели большего размера не с чем сопоставить.

    python scripts/join_markup65.py --out data/judge_9b_markup65.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def читать(путь: str) -> list[dict]:
    п = Path(путь)
    if not п.exists():
        return []
    return [json.loads(s) for s in п.read_text(encoding="utf-8").splitlines() if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--эталон", default="data/markup65_for_judge.jsonl")
    ap.add_argument("--прогоны", nargs="*",
                    default=["data/judge_nl_candidates.jsonl", "data/judge_nl_clean.jsonl"])
    ap.add_argument("--out", default="data/judge_9b_markup65.jsonl")
    a = ap.parse_args()

    эталон = {z["id"]: z for z in читать(a.эталон)}
    вердикты: dict[str, dict] = {}
    условия: set[tuple] = set()
    for путь in a.прогоны:
        for z in читать(путь):
            # Эталон может быть свёрнут по ходу, а прогоны сделаны по строке на требование:
            # `airline__22::t11#nl11` и `#nl12` — один и тот же вход, и вердикт у них
            # одинаковый по построению. Берём первый попавшийся и не считаем дважды.
            ключ = z["id"] if z["id"] in эталон else z["id"].split("#")[0]
            if ключ in эталон and ключ not in вердикты:
                вердикты[ключ] = z
                условия.add((z.get("модель"), z.get("окно")))

    if len(условия) > 1:
        # Смешанный файл читается как один замер и врёт `[урок 31.08.2026]`.
        print(f"[ОТКАЗ] прогоны сделаны при разных условиях: {sorted(условия)}")
        return 2

    строки = []
    for id_, э in эталон.items():
        в = вердикты.get(id_)
        if в is None:
            continue
        строки.append({**в, "label": э["label"], "ответ": э["response"],
                       "вид": э.get("вид", ""), "единогласно": э.get("единогласно")})

    Path(a.out).write_text("".join(json.dumps(z, ensure_ascii=False) + "\n" for z in строки),
                           encoding="utf-8")
    поз = sum(z["label"] for z in строки)
    м, о = next(iter(условия)) if условия else (None, None)
    print(f"записано {len(строки)} строк в {a.out}: положительных {поз}, "
          f"отрицательных {len(строки) - поз}; условия: модель {м}, окно {о}")
    if len(строки) < len(эталон):
        print(f"⚠ без вердикта осталось {len(эталон) - len(строки)} строк эталона")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

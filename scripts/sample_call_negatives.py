"""Выборка надёжных отрицательных С ВЫЗОВОМ инструмента — страта, которая не измерена.

Зачем. Доля ложных срабатываний судьи в страте с вызовом всё это время бралась с выданного
сэмпла, где в ней **два** отрицательных примера `[§0в]`, а в боевом наборе их 134. Из этих
двух судья обвинял оба, и число 1.000 попадало во все проекции. `[расчёт 01.09.2026]` Одно
это число двигает итоговый F1 с **0.283 до 0.445** — то есть с «глубоко ниже пола 0.393»
до «выше пола». Мерить его на двух точках нельзя.

Материал есть: в `corpus_v2.csv` лежат **88 490** ходов с меткой 0, вызовом инструмента
и источником `solved_no_gate` — то есть из симуляций, где решены ВСЕ требования. Это
надёжные отрицательные: их чистота следует из разметки бенчмарка, а не из наших гейтов
(круговая проверка, стоившая проекту трёх ложных выводов, здесь исключена по построению).

Запускать **на мини-ПК**: индекс хранит ссылки, а текст восстанавливается из траекторий,
которые лежат там (5.7 ГБ). Результат — обычный jsonl для `judge_sanity.py`.

    python3 scripts/sample_call_negatives.py --index ~/sverka/tau2/corpus_v2.csv \
        --data ~/sverka/tau2/data --n 120 --out ~/sverka/call_negatives.jsonl
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.tau2 import render_by_ref  # noqa: E402

СИД = 20260901


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True, help="каталог траекторий")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=120_000,
                    help="потолок длины промпта: сверхдлинные съедают прогон и всё равно "
                         "усекаются подачей")
    a = ap.parse_args()

    csv.field_size_limit(10 ** 9)
    годные = []
    with open(a.index, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["label"] == "0" and r["has_tool_calls"] == "1"
                    and r["label_source"] == "solved_no_gate"
                    and int(r["prompt_chars"] or 0) <= a.max_chars):
                годные.append(r)
    print(f"надёжных отрицательных с вызовом: {len(годные)}")

    rnd = random.Random(СИД)
    rnd.shuffle(годные)

    # Домены выравниваются: в корпусе они представлены очень неравномерно, а боевой набор
    # делится поровну на четыре `[§данные]`. Без выравнивания замер описывал бы тот домен,
    # которого в корпусе больше всего.
    по_домену: dict[str, list] = collections.defaultdict(list)
    for r in годные:
        по_домену[r["domain"]].append(r)
    print("  по доменам: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(по_домену.items())))
    на_домен = max(1, a.n // max(1, len(по_домену)))
    взято = [r for д in sorted(по_домену) for r in по_домену[д][:на_домен]]

    строки, пропущено = [], 0
    for r in взято:
        путь = Path(a.data) / Path(r["source_file"]).name
        if not путь.exists():
            путь = Path(r["source_file"])
        пара = render_by_ref(путь, int(r["sim_index"]), int(r["turn_idx"]))
        if пара is None:
            пропущено += 1
            continue
        prompt, response = пара
        if "TOOL_CALL" not in response:
            # Страховка: страта определяется по РЕНДЕРУ, а не по колонке индекса. Если они
            # разойдутся, замер молча уедет в другую страту.
            пропущено += 1
            continue
        строки.append({"id": r["id"], "prompt": prompt, "response": response,
                       "label": 0, "domain": r["domain"]})

    Path(a.out).write_text("".join(json.dumps(z, ensure_ascii=False) + "\n" for z in строки),
                           encoding="utf-8")
    print(f"записано {len(строки)} строк в {a.out}, пропущено {пропущено}")
    print("  домены: " + ", ".join(f"{k} {v}" for k, v in
                                   sorted(collections.Counter(z["domain"] for z in строки).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

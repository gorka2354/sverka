"""Парный набор лёгких ходов из пула с привязкой по позиции: кандидаты против контроля.

Зачем он нужен рядом с размеченным набором. В размеченном наборе слой лёгких ходов —
тот самый, где лежат 91% боевых, — состоит из **шести** положительных `[inc-4]`. На шести
точках разница 0/6 против 2/6 незначима, и замер способен только опровергнуть надежду,
но не подтвердить её. Пул `mine_nl_free` даёт 119 лёгких кандидатов и 66 лёгких контрольных
без всякой ручной разметки, потому что привязка там идёт по позиции хода, а не по якорям.

⚠️ **Метки здесь шумные, и это не оговорка, а свойство набора.** Кандидат — ход симуляции,
где требование провалено, но провалено оно могло быть и не на этом ходе; часть кандидатов
чиста. Контроль — ход симуляции, где требование выполнено, и он надёжен. Поэтому:

- **полнота на кандидатах — оценка СНИЗУ** для настоящей полноты;
- **доля срабатываний на контроле — честная оценка доли ложных**;
- сравнивать надо ГРУППЫ между собой, а абсолютную полноту читать нельзя.

Отбор внутри слоя лёгких: доля лёгких в пуле (0.26) не равна боевой (0.91), но ставка,
посчитанная среди лёгких, переносится на боевые лёгкие. Смещение долей слоёв лечится
взвешиванием, смещение ставки внутри слоя — ничем, поэтому мерить надо внутри.

    python scripts/build_free_pairs.py --n 40 --out data/free_pairs_light.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.units import извлечь  # noqa: E402

СИД = 20260901


def читать(путь: str) -> list[dict]:
    return [json.loads(s) for s in Path(путь).read_text(encoding="utf-8").splitlines()
            if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--поз", default="data/nl_free_pos_ru.jsonl")
    ap.add_argument("--ктл", default="data/nl_free_ctl_ru.jsonl")
    ap.add_argument("--n", type=int, default=40, help="сколько взять в каждую группу")
    ap.add_argument("--единиц", type=int, default=2, help="потолок тяжести хода")
    ap.add_argument("--out", default="data/free_pairs_light.jsonl")
    a = ap.parse_args()

    def лёгкие(строки):
        # Один ход на симуляцию: пул строится по одному ходу, но подстраховаться дешевле,
        # чем потом объяснять дубли в метрике `[дефект 31.08.2026]`.
        видели, годные = set(), []
        for z in строки:
            ход = z["id"].split("#")[0]
            if ход in видели:
                continue
            if len(извлечь(z.get("response", ""))) <= a.единиц:
                видели.add(ход)
                годные.append({**z, "id": ход})
        return годные

    поз, ктл = лёгкие(читать(a.поз)), лёгкие(читать(a.ктл))
    rnd = random.Random(СИД)
    rnd.shuffle(поз); rnd.shuffle(ктл)
    взято = ([{**z, "label": 1, "группа": "кандидат"} for z in поз[: a.n]] +
             [{**z, "label": 0, "группа": "контроль"} for z in ктл[: a.n]])
    rnd.shuffle(взято)

    строки = [{"id": z["id"], "prompt": z["prompt"], "response": z["response"],
               "label": z["label"], "domain": z.get("домен", ""), "группа": z["группа"]}
              for z in взято]
    Path(a.out).write_text("".join(json.dumps(z, ensure_ascii=False) + "\n" for z in строки),
                           encoding="utf-8")
    д = sorted(len(z["response"]) for z in строки)
    print(f"записано {len(строки)} строк в {a.out}: кандидатов "
          f"{sum(z['label'] for z in строки)}, контроля {sum(1 for z in строки if not z['label'])}")
    print(f"  доступно лёгких: кандидатов {len(поз)}, контроля {len(ктл)}")
    print(f"  медиана ответа {д[len(д) // 2]} символов "
          f"(боевые лёгкие: 348) — профиль близок, если числа одного порядка")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

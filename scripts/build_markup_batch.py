"""Собрать партию карточек для разметки: новые вперемешку со старыми, уже размеченными.

Зачем подмешивать старые. Новые метки лягут в один набор со старыми `[§0е]`, а размечались
те другим проходом, промпт которого не сохранился. Складывать две разметки, не зная, дают
ли они одно и то же, — это построить набор, в котором часть меток означает не то же, что
другая часть. Поэтому в партию идут 15 карточек с уже известным вердиктом: если разметчик
воспроизведёт их с высоким согласием, наборы совместимы, и старые метки можно оставить как
есть; если нет — весь набор придётся переразмечать одним проходом.

Карточки перенумерованы заново и перемешаны фиксированным сидом: разметчик не должен
отличать новые от контрольных, иначе контроль не контроль.

    python scripts/build_markup_batch.py --out data/markup_batch3.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

СИД = 20260901


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--новые", default="data/markup_cards3.json")
    ap.add_argument("--старые", default="data/markup_cards2.json")
    ap.add_argument("--старая-разметка", default="data/markup_machine2.json")
    ap.add_argument("--контроль", type=int, default=15)
    ap.add_argument("--out", default="data/markup_batch3.json")
    a = ap.parse_args()

    новые = json.loads(Path(a.новые).read_text(encoding="utf-8"))
    старые = json.loads(Path(a.старые).read_text(encoding="utf-8"))
    разметка = json.loads(Path(a.старая_разметка).read_text(encoding="utf-8"))

    rnd = random.Random(СИД)
    # Контроль берётся только из единогласных вердиктов: карточка, на которой прошлый
    # проход сам разошёлся 2:1, ничего не проверяет — по ней и согласие не с чем сравнивать.
    пригодны = [к for к in старые
                if len(set(разметка.get(str(к["n"]), {}).get("голоса", ["x", "y"]))) == 1]
    rnd.shuffle(пригодны)
    контроль = пригодны[: a.контроль]

    партия = []
    for к in новые:
        партия.append({**к, "_источник": "новая", "_старый_n": None})
    for к in контроль:
        партия.append({**к, "_источник": "контроль", "_старый_n": к["n"]})
    rnd.shuffle(партия)

    ключ = {}
    выдача = []
    for i, к in enumerate(партия, 1):
        ключ[str(i)] = {
            "источник": к["_источник"],
            "id": к["id"],
            "прошлый_вердикт": (разметка.get(str(к["_старый_n"]), {}).get("majority")
                                if к["_старый_n"] else None),
        }
        выдача.append({"n": i, "домен": к["домен"], "запрос": к["запрос"],
                       "ответ": к["ответ"], "срез": к["срез"], "требование": к["требование"]})

    Path(a.out).write_text(json.dumps(выдача, ensure_ascii=False, indent=1), encoding="utf-8")
    п = Path(a.out).with_name(Path(a.out).stem + "_ключ.json")
    п.write_text(json.dumps(ключ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"партия: {len(выдача)} карточек → {a.out} "
          f"({Path(a.out).stat().st_size // 1024} КБ)")
    print(f"  новых {sum(1 for v in ключ.values() if v['источник'] == 'новая')}, "
          f"контрольных {sum(1 for v in ключ.values() if v['источник'] == 'контроль')}")
    print(f"  ключ (разметчику не показывать): {п}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

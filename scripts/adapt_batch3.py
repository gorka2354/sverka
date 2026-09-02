"""Привести партию `markup_batch3` к формату, который читает сборщик набора.

Партия перенумерована и перемешана, а `id` из неё намеренно вынут — иначе разметчик отличил
бы контрольные карточки от новых. Поэтому склейка идёт через ключ партии, и здесь она
оформляется в те же две таблицы, что даёт `build_markup_cards.py`: карточки с `id` и вид,
и разметка по тем же номерам.

Берутся ТОЛЬКО новые карточки. Контрольные пятнадцать своё дело сделали — показали,
что новая разметка воспроизводит прошлую (15/15) — и второй раз в набор идти не должны.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ключ", default="data/markup_batch3_ключ.json")
    ap.add_argument("--разметка", default="data/markup_machine3.json")
    ap.add_argument("--вид", default="data/markup_cards3_ключ.json")
    ap.add_argument("--карточки3", default="data/markup_cards3.json")
    ap.add_argument("--out-карточки", default="data/markup_cards3_final.json")
    ap.add_argument("--out-разметка", default="data/markup_machine3_final.json")
    a = ap.parse_args()

    ключ = json.loads(Path(a.ключ).read_text(encoding="utf-8"))
    разметка = json.loads(Path(a.разметка).read_text(encoding="utf-8"))
    вид_по_n = json.loads(Path(a.вид).read_text(encoding="utf-8"))
    карточки3 = json.loads(Path(a.карточки3).read_text(encoding="utf-8"))
    вид_по_id = {к["id"]: вид_по_n.get(str(к["n"]), "") for к in карточки3}

    карточки, машина = [], {}
    for n, мета in ключ.items():
        if мета["источник"] != "новая":
            continue
        м = разметка.get(n)
        if м is None:
            continue
        вид = вид_по_id.get(мета["id"], "")
        карточки.append({"n": int(n), "id": мета["id"], "домен": "", "вид": вид})
        машина[n] = {"majority": м["majority"], "голоса": м["голоса"], "вид": вид}

    Path(a.out_карточки).write_text(json.dumps(карточки, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    Path(a.out_разметка).write_text(json.dumps(машина, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    да = sum(1 for v in машина.values() if v["majority"] == "da")
    чистых = sum(1 for v in машина.values() if v["вид"] == "чистая")
    print(f"новых карточек: {len(карточки)} → {a.out_карточки}")
    print(f"  «да» большинством: {да}; заведомо чистых (метка 0 независимо от вердикта): "
          f"{чистых}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

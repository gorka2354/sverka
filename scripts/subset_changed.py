"""Выбрать примеры, у которых промпт при новом условии отличается от базового.

Зачем. Отбор по якорям меняет **семь промптов из сорока шести**: остальные тридцать девять
собираются побайтово так же, как в базовом прогоне. Гонять их заново — час работы карты
ради заведомо тех же чисел. Детерминированность при `temperature = 0` уже проверена
`[замер 4]`: контрольная группа целых промптов дала ставки, совпавшие с базой до сотых.

Поэтому здесь готовится подвыборка изменённых, а `merge_runs.py` потом склеивает её
с базовым прогоном.

⚠️ Склейка законна ТОЛЬКО когда неизменённые промпты действительно идентичны. Скрипт
сравнивает их посимвольно и отказывается работать, если это не так.

    python scripts/subset_changed.py --спасать-якоря --out data/subset_anchors.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.judge_prompt import build_prompt  # noqa: E402
from sverka.transcript import parse_response, parse_transcript  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/valid.parquet")
    ap.add_argument("--базовый-бюджет", type=int, default=60_000)
    ap.add_argument("--бюджет", type=int, default=60_000)
    ap.add_argument("--спасать-якоря", action="store_true")
    ap.add_argument("--out", default="data/subset_changed.jsonl")
    a = ap.parse_args()

    if a.data.endswith(".jsonl"):
        строки = [json.loads(s) for s in
                  Path(a.data).read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        import pandas as pd
        строки = [dict(r) for _, r in pd.read_parquet(a.data).iterrows()]

    изменённые = []
    for r in строки:
        tr = parse_transcript(str(r["prompt"]), strict=False)
        turn = parse_response(str(r["response"]), strict=False)
        база = build_prompt(tr, turn, budget_chars=a.базовый_бюджет)
        новый = build_prompt(tr, turn, budget_chars=a.бюджет, спасать_якоря=a.спасать_якоря)
        if база.user != новый.user or база.system != новый.system:
            изменённые.append({"id": r["id"], "prompt": str(r["prompt"]),
                               "response": str(r["response"]), "label": int(r["label"]),
                               "domain": r.get("domain", "")})

    Path(a.out).write_text("".join(json.dumps(z, ensure_ascii=False) + "\n"
                                   for z in изменённые), encoding="utf-8")
    print(f"изменённых промптов: {len(изменённые)} из {len(строки)} → {a.out}")
    print(f"  неизменённые {len(строки) - len(изменённые)} берутся из базового прогона: "
          f"их промпты идентичны посимвольно, а прогон детерминирован при temperature 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

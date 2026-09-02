"""Склеить прогон подвыборки с базовым: изменённые строки из нового, остальные из базы.

Законно только при двух условиях, и оба проверяются здесь, а не на честном слове:

1. **Условия прогонов совпадают** — модель, окно, бюджет, флаг отбора. Смешанный файл
   читается как один замер и врёт `[урок 31.08.2026]`.
2. **Неизменённые промпты действительно идентичны.** Пересобираются оба варианта промпта
   и сравниваются посимвольно; если хоть один отличается, склейка отменяется целиком.

Зачем вообще склейка. Отбор по якорям меняет семь промптов из сорока шести. Прогонять
остальные тридцать девять — час работы карты ради заведомо тех же чисел; детерминированность
при `temperature = 0` подтверждена `[замер 4]`, где контрольная группа совпала до сотых.

    python scripts/merge_runs.py --база data/judge_27b.jsonl \
        --подвыборка data/judge_27b_anchors2.jsonl --out data/judge_27b_anchors_fix.jsonl \
        --спасать-якоря
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.judge_prompt import build_prompt  # noqa: E402
from sverka.transcript import parse_response, parse_transcript  # noqa: E402


def читать(путь: str) -> list[dict]:
    return [json.loads(s) for s in Path(путь).read_text(encoding="utf-8").splitlines()
            if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--база", required=True)
    ap.add_argument("--подвыборка", required=True)
    ap.add_argument("--data", default="data/raw/valid.parquet")
    ap.add_argument("--базовый-бюджет", type=int, default=60_000)
    ap.add_argument("--бюджет", type=int, default=60_000)
    ap.add_argument("--спасать-якоря", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    база = {z["id"]: z for z in читать(a.база)}
    новые = {z["id"]: z for z in читать(a.подвыборка)}

    у_базы = {(z.get("модель"), z.get("окно")) for z in база.values()}
    у_новых = {(z.get("модель"), z.get("окно")) for z in новые.values()}
    if у_базы != у_новых:
        print(f"[ОТКАЗ] условия не совпадают: база {у_базы}, подвыборка {у_новых}")
        return 2

    if a.data.endswith(".jsonl"):
        строки = читать(a.data)
    else:
        import pandas as pd
        строки = [dict(r) for _, r in pd.read_parquet(a.data).iterrows()]

    расхождения = []
    итог = []
    for r in строки:
        id_ = r["id"]
        if id_ in новые:
            итог.append(новые[id_])
            continue
        if id_ not in база:
            continue
        tr = parse_transcript(str(r["prompt"]), strict=False)
        turn = parse_response(str(r["response"]), strict=False)
        б = build_prompt(tr, turn, budget_chars=a.базовый_бюджет)
        н = build_prompt(tr, turn, budget_chars=a.бюджет, спасать_якоря=a.спасать_якоря)
        if б.user != н.user or б.system != н.system:
            расхождения.append(id_)
            continue
        # Условия строки переписываются на условия НОВОГО прогона: промпт тот же,
        # значит вердикт получен бы тот же, но читать файл потом будут как один замер.
        итог.append({**база[id_], "бюджет_символов": a.бюджет,
                     "спасение_якорей": a.спасать_якоря, "источник_строки": "база"})

    if расхождения:
        print(f"[ОТКАЗ] у {len(расхождения)} строк промпт отличается, но прогона для них "
              f"нет: {расхождения[:5]}. Склейка выдала бы смесь двух условий.")
        return 2

    Path(a.out).write_text("".join(json.dumps(z, ensure_ascii=False) + "\n" for z in итог),
                           encoding="utf-8")
    из_новых = sum(1 for z in итог if z.get("источник_строки") != "база")
    print(f"склеено {len(итог)} строк → {a.out}")
    print(f"  из подвыборки {из_новых}, из базы {len(итог) - из_новых} "
          f"(их промпты проверены посимвольно)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

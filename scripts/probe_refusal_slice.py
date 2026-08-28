"""Замер: можно ли выделить ложный отказ механически, без модели и без запуска окружения.

Почему это важно. Ложный отказ — второй по массовости класс в выданном сэмпле (5 из 23
позитивов) и единственный, под который нет ни детерминированной проверки, ни опубликованного
метода. Он же живёт преимущественно в страте «ответ без вызова инструмента», где доля
позитива 0.09 и где сосредоточена вся содержательная работа.

Идея замера. В публичных траекториях τ²-bench есть два независимых сигнала:
`reward_info.reward` (решена ли задача) и `reward_info.action_checks` (по каждому ОЖИДАЕМОМУ
действию известно, сматчилось оно или нет). Если агент перевёл разговор на оператора,
а ожидаемые действия остались невыполненными и задача провалена — он сдался там, где
задача была выполнима. Если же перевод случился, а все ожидаемые действия выполнены
и задача решена — перевод был обоснован.

`[замер 28.08.2026, 6 файлов]` Контраст резкий:

    перевод + reward=0 + есть невыполненные действия   254   ← кандидат в ложный отказ
    перевод + reward=0 + все действия выполнены          7
    перевод + reward=1 + все действия выполнены        252   ← обоснованный отказ
    перевод + reward=1 + есть невыполненные              15

То есть один и тот же поступок (перевод на оператора) делится на два класса почти без
пересечения. Это даёт и позитивы, и СЛОЖНЫЕ НЕГАТИВЫ — без них детектор выучит «отказ
всегда галлюцинация», что на выданном сэмпле прямо опровергается: там два из шести ходов
с переводом размечены нулём.

Границы. Сигнал слабый и косвенный: ожидаемое действие могло остаться невыполненным
по другой причине, а задача — провалиться на следующем ходу. Поэтому метка называется
кандидатом, а не эталоном, и требует ручной вычитки на выборке.

    python scripts/probe_refusal_slice.py --data ~/sverka/tau2/data --limit-files 6
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REFUSAL_TOOLS = ("transfer_to_human_agents", "request_human_agent_transfer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit-files", type=int, default=6)
    ap.add_argument("--max-mb", type=float, default=60.0)
    args = ap.parse_args()

    files = [f for f in sorted(Path(args.data).expanduser().glob("*.json"))
             if f.stat().st_size <= args.max_mb * 1e6][: args.limit_files]
    print(f"файлов: {len(files)}")

    buckets: collections.Counter = collections.Counter()
    turns: collections.Counter = collections.Counter()
    examples: list[str] = []

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for sim in data.get("simulations") or []:
            ri = sim.get("reward_info") or {}
            reward = float(ri.get("reward") or 0.0)
            msgs = sim.get("messages") or []

            for m in msgs:
                if str(m.get("role", "")).lower() != "assistant":
                    continue
                calls = m.get("tool_calls") or []
                if not calls:
                    turns["текстовый ход агента"] += 1
                    turns["текстовый, решённая" if reward == 1 else "текстовый, проваленная"] += 1
                elif any(c.get("name") in REFUSAL_TOOLS for c in calls):
                    turns["ход с переводом на оператора"] += 1
                else:
                    turns["ход с обычным вызовом"] += 1

            if not any(c.get("name") in REFUSAL_TOOLS
                       for m in msgs for c in (m.get("tool_calls") or [])):
                continue
            checks = ri.get("action_checks") or []
            missed = [a for a in checks if not a.get("action_match")]
            key = ("проваленная" if reward == 0 else "решённая") + (
                ", есть невыполненные действия" if missed else ", все действия выполнены"
            )
            buckets[key] += 1
            if reward == 0 and missed and len(examples) < 5:
                names = [a.get("action", {}).get("name", "?") for a in missed][:4]
                examples.append(
                    f"задача {sim.get('task_id')}: ожидалось {len(checks)}, "
                    f"не выполнено {len(missed)} — {names}"
                )

    print("\nходы агента по типам:")
    for k, v in turns.most_common():
        print(f"  {k:34s} {v:7d}")

    print("\nсимуляции, где агент переводил на оператора:")
    for k in sorted(buckets):
        print(f"  {k:44s} {buckets[k]:5d}")

    pos = buckets["проваленная, есть невыполненные действия"]
    neg = buckets["решённая, все действия выполнены"]
    total = sum(buckets.values())
    if total:
        print(
            f"\nкандидатов в ложный отказ: {pos} ({100*pos/total:.0f}% переводов)\n"
            f"обоснованных отказов (сложные негативы): {neg} ({100*neg/total:.0f}%)"
        )
        print("Разделение почти без пересечения — сигнал пригоден как слабая метка.")

    print("\nпримеры кандидатов:")
    for e in examples:
        print(f"  {e}")
    print(
        "\n⚠️ Метка слабая: ожидаемое действие могло остаться невыполненным по другой причине,\n"
        "   а задача — провалиться на следующем ходу. Нужна ручная вычитка на выборке."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

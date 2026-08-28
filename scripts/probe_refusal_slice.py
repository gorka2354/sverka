"""Замер: как выделить ложный отказ, и почему первая формулировка была тавтологичной.

Ложный отказ — второй по массовости класс в выданном сэмпле (5 из 23 позитивов) и
единственный, под который нет ни детерминированной проверки, ни опубликованного метода.

## Первая формулировка и её опровержение

Правило было: «агент перевёл на оператора + задача провалена + ожидаемые действия
не выполнены». Разбиение выглядело резким (254 кандидата против 7, и 252 обоснованных
отказа против 15) и было доложено как находка. **Оно в основном тавтологично.**

`[замер 28.08.2026]` В τ²-bench итоговый reward складывается из компонент, и
`evaluator_action.py` считает `reward = 1.0 if all_expectations_met else 0.0`. То есть
«действия не выполнены» — это одна из причин `reward = 0`, а не независимый признак.
Данные подтверждают: среди ВСЕХ симуляций (не только с переводом) невыполненные действия
есть у **86.8% провалов** против 11.7% успехов. Значит контраст 254/7 в основном пересказывает
«провал против успеха», а роль перевода в нём почти не видна.

## Что несёт сигнал на самом деле

Независимая информация приходит от вопроса **просил ли пользователь человека до перевода**.
`[замер 28.08.2026, 6 файлов]` Доля «провал + невыполненные действия» среди переводов:

    домен      пользователь НЕ просил   пользователь просил
    airline        3/14  = 21%             10/89 = 11%
    retail         4/9   = 44%              8/46 = 17%
    telecom      188/293 = 64%             41/77 = 53%

Условие поднимает долю во всех трёх доменах, а заодно видно главное: подавляющее большинство
переводов происходит ПОСЛЕ просьбы пользователя (89 из 103 в airline, 46 из 55 в retail).
Такие переводы обоснованы, и именно они дают сложные негативы — без них детектор выучит
«отказ всегда галлюцинация», что на выданном сэмпле прямо опровергается: два из шести ходов
с переводом размечены нулём.

## Итоговое правило

Кандидат в ложный отказ = перевод на оператора **И** пользователь его не просил **И**
остались невыполненные ожидаемые действия. Сложный негатив = перевод после просьбы
пользователя при выполненных действиях.

Метка остаётся слабой и требует ручной вычитки на выборке: ожидаемое действие могло
остаться невыполненным по другой причине, а задача — провалиться на следующем ходу.

    python scripts/probe_refusal_slice.py --data ~/sverka/tau2/data --limit-files 6
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REFUSAL_TOOLS = ("transfer_to_human_agents", "request_human_agent_transfer")

# Просьба пользователя соединить с человеком. Список намеренно широкий: пропустить такую
# просьбу опаснее, чем поймать лишнюю — пропуск превращает обоснованный перевод в «ложный
# отказ» и отравляет позитивы.
ASKED_FOR_HUMAN = re.compile(
    r"human|real person|representative|manager|supervisor|operator|agent\b|escalat|transfer",
    re.I,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit-files", type=int, default=6)
    ap.add_argument("--max-mb", type=float, default=60.0)
    args = ap.parse_args()

    files = [f for f in sorted(Path(args.data).expanduser().glob("*.json"))
             if f.stat().st_size <= args.max_mb * 1e6][: args.limit_files]
    print(f"файлов: {len(files)}")

    turns: collections.Counter = collections.Counter()
    tautology: collections.Counter = collections.Counter()
    split: collections.Counter = collections.Counter()
    examples: list[str] = []

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        domain = ((data.get("info") or {}).get("environment_info") or {}).get("domain_name", "?")

        for sim in data.get("simulations") or []:
            ri = sim.get("reward_info") or {}
            reward = float(ri.get("reward") or 0.0)
            msgs = sim.get("messages") or []
            checks = ri.get("action_checks") or []
            missed = [a for a in checks if not a.get("action_match")]

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

            # Проверка тавтологичности: связь «провал» и «невыполненные действия»
            # на ВСЕХ симуляциях, независимо от перевода.
            if checks:
                tautology[("провал" if reward == 0 else "успех")
                          + (", есть невыполненные" if missed else ", всё выполнено")] += 1

            # Индекс первого перевода и была ли до него просьба пользователя.
            idx = None
            for i, m in enumerate(msgs):
                if str(m.get("role", "")).lower() == "assistant" and any(
                    c.get("name") in REFUSAL_TOOLS for c in (m.get("tool_calls") or [])
                ):
                    idx = i
                    break
            if idx is None:
                continue
            asked = any(
                ASKED_FOR_HUMAN.search(str(m.get("content") or ""))
                for m in msgs[:idx]
                if str(m.get("role", "")).lower() == "user"
            )
            split[(domain, asked, bool(missed) and reward == 0)] += 1

            if not asked and missed and reward == 0 and len(examples) < 5:
                names = [a.get("action", {}).get("name", "?") for a in missed][:4]
                examples.append(
                    f"{domain} / задача {sim.get('task_id')}: пользователь человека не просил, "
                    f"не выполнено {len(missed)} из {len(checks)} — {names}"
                )

    print("\nходы агента по типам:")
    for k, v in turns.most_common():
        print(f"  {k:34s} {v:7d}")

    print("\nтавтологичность первой формулировки (все симуляции с action_checks):")
    fail = tautology["провал, есть невыполненные"] + tautology["провал, всё выполнено"]
    ok = tautology["успех, есть невыполненные"] + tautology["успех, всё выполнено"]
    for k in sorted(tautology):
        print(f"  {k:34s} {tautology[k]:6d}")
    if fail and ok:
        pf = tautology["провал, есть невыполненные"] / fail
        po = tautology["успех, есть невыполненные"] / ok
        print(f"  доля провалов с невыполненными: {pf:.3f}; успехов: {po:.3f}")
        if pf > 0.8:
            print("  → связь почти детерминированная: «провал + невыполнено» ≈ «провал».")
            print("    Первая формулировка правила независимого сигнала не несла.")

    print("\nпереводы в разрезе «просил ли пользователь человека до перевода»:")
    print(f"  {'домен':<20} {'НЕ просил':<22} {'просил':<22}")
    for domain in sorted({k[0] for k in split}):
        row = []
        for asked in (False, True):
            hit = split[(domain, asked, True)]
            tot = hit + split[(domain, asked, False)]
            row.append(f"{hit}/{tot} = {100*hit/tot:.0f}%" if tot else "нет данных")
        print(f"  {domain:<20} {row[0]:<22} {row[1]:<22}")
    print("  Колонка «НЕ просил» выше — условие несёт сигнал сверх reward.")
    print("  Колонка «просил» и есть источник СЛОЖНЫХ НЕГАТИВОВ: перевод там обоснован.")

    print("\nпримеры кандидатов в ложный отказ:")
    for e in examples:
        print(f"  {e}")
    print(
        "\n⚠️ Метка слабая. Ожидаемое действие могло остаться невыполненным по другой причине,\n"
        "   задача — провалиться на следующем ходу, а регулярка просьбы о человеке ловит\n"
        "   и слово «agent» в нейтральном смысле. Нужна ручная вычитка на выборке."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

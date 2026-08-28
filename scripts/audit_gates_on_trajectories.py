"""T7 — аудит точности гейтов на публичных траекториях τ²-bench.

Зачем это существует. На выданных 46 примерах точность гейтов не измеряется в принципе:
все они требуют вызова инструмента в ответе, а в этой страте всего два отрицательных
примера, и оба — одиночный `transfer_to_human_agents`, на который ни один гейт сработать
не может. Достижимых отрицательных ноль. Здесь набор, где они достижимы.

**Прокси-оракул и его честные границы.** Разметки на уровне хода в публичных траекториях
нет. Берутся симуляции с `reward == 1.0` — задача решена, конечное состояние базы совпало
с эталоном. Ходы таких симуляций в подавляющем большинстве корректны, поэтому срабатывание
гейта читается как вероятное ложное. Оракул шумный: ход мог нарушить процедурное правило,
не изменив итогового состояния базы, и такое срабатывание было бы верным. Поэтому вывод
формулируется как «доля срабатываний на решённых задачах», а не как «точность», и каждое
срабатывание печатается с обоснованием, чтобы его можно было прочитать глазами.

    python scripts/audit_gates_on_trajectories.py --data ~/sverka/tau2/data --limit-files 12
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.gates import apply_gates, load_known_user_tools  # noqa: E402
from sverka.tau2 import iter_turns  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="каталог с файлами траекторий")
    ap.add_argument("--limit-files", type=int, default=12)
    ap.add_argument("--max-sims", type=int, default=None, help="ограничить симуляций на файл")
    ap.add_argument("--max-mb", type=float, default=80.0, help="пропускать файлы крупнее")
    ap.add_argument("--show", type=int, default=12, help="сколько срабатываний распечатать")
    args = ap.parse_args()

    user_tools = load_known_user_tools()
    files = sorted(Path(args.data).glob("*.json"))
    files = [f for f in files if f.stat().st_size <= args.max_mb * 1e6][: args.limit_files]
    print(f"файлов к разбору: {len(files)}")

    total = 0
    fired = 0
    by_gate: collections.Counter = collections.Counter()
    by_domain_seen: collections.Counter = collections.Counter()
    by_domain_fired: collections.Counter = collections.Counter()
    examples: list[str] = []

    for f in files:
        n_file = 0
        try:
            for tt in iter_turns(f, only_reward=1.0, max_sims=args.max_sims):
                total += 1
                n_file += 1
                by_domain_seen[tt.domain] += 1
                rep = apply_gates(
                    tt.transcript, tt.turn, user_tools=user_tools.get(tt.domain, set())
                )
                if rep.silenced:
                    continue
                if rep.fired:
                    fired += 1
                    by_domain_fired[tt.domain] += 1
                    for h in rep.hits:
                        by_gate[h.gate] += 1
                    if len(examples) < args.show:
                        examples.append(f"{tt.uid} [{tt.source[:40]}] -> {rep}")
        except Exception as e:  # noqa: BLE001 — один битый файл не должен ронять аудит
            print(f"  ПРОПУЩЕН {f.name}: {type(e).__name__}: {e}")
            continue
        print(f"  {f.name[:70]:70s} ходов с вызовом при reward=1.0: {n_file}")

    print()
    print("=" * 78)
    print(f"ВСЕГО ходов агента с вызовом инструмента в решённых задачах: {total}")
    if total == 0:
        print("нечего мерить")
        return 1
    print(f"из них сработал хотя бы один гейт: {fired} ({100*fired/total:.2f}%)")
    print()
    print("срабатывания по гейтам:")
    for g, n in by_gate.most_common():
        print(f"  {g:22s} {n:5d}  ({100*n/total:.2f}% от всех ходов)")
    print()
    print("по доменам:")
    for d in sorted(by_domain_seen):
        seen, fr = by_domain_seen[d], by_domain_fired[d]
        print(f"  {d:20s} ходов={seen:5d}  сработало={fr:4d}  ({100*fr/seen if seen else 0:.2f}%)")
    print()
    print("примеры срабатываний (читать глазами — оракул шумный):")
    for e in examples:
        print(f"  {e[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

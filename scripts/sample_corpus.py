"""Отбор согласованной выборки из индекса корпуса и материализация текста.

Зачем нужен именно СОГЛАСОВАННЫЙ отбор. Приёмочная проба завернула корпус v2: позитивы,
добытые гейтами, отделяются от негативов по домену (перекос ×411), по наличию вызова
инструмента (∞ — в текстовой страте позитивов нет вовсе), по номеру хода (×31) и по длине
ответа (×15). Если из такого индекса брать случайную выборку, все перекосы поедут в неё
целиком, и любой замер на ней будет мерить их, а не задачу.

Поэтому отбор идёт по стратам `(домен, есть ли вызов, корзина номера хода)`: в каждой
страте берётся одинаковое число строк каждого класса, сколько получится. Что не набралось —
печатается явно, а не замалчивается.

## Почему выборка маленькая, а не «весь пул»

Пул кандидатов — 134 978 строк. Судья на локальной 9B считает пример около 40 секунд
`[замер E1b]`, то есть весь пул это полтора года. Смысл выборки не в объёме, а в том,
чтобы измерить две величины, которых у нас нет:

- **долю ложных срабатываний судьи на надёжных негативах.** На выданном сэмпле она
  неизмерима: в нужной страте всего два достижимых отрицательных примера. Здесь их
  176 622, и можно брать сколько нужно.
- **выход по кандидатам** — какую долю пула судья помечает. Вместе с первой величиной
  это даёт оценку, сколько среди помеченных настоящих позитивов.

## Материализация

Индекс хранит ссылки, а не текст. Материализация идёт по файлам: строки группируются
по `source_file`, и каждый файл читается ОДИН раз. Наивный вызов `render_by_ref` на строку
сканировал бы файл заново каждый раз — на banking-файлах по 250 МБ это часы.

    python scripts/sample_corpus.py --index corpus_v2.csv --data ~/sverka/tau2/data \\
        --out sample.csv --per-stratum 4 --classes 0 -1
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from sverka.tau2 import iter_turns, render_example  # noqa: E402


def turn_bucket(v: str) -> str:
    n = int(v or 0)
    return "t<5" if n < 5 else ("t5-15" if n < 15 else "t>=15")


def stratum(r: dict) -> tuple[str, str, str]:
    return (r["domain"], r["has_tool_calls"], turn_bucket(r["turn_idx"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True, help="каталог с файлами траекторий")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stratum", type=int, default=4,
                    help="сколько строк каждого класса брать из каждой страты")
    ap.add_argument("--classes", nargs="+", default=["0", "-1"],
                    help="какие метки отбирать: 0 негативы, 1 позитивы, -1 кандидаты")
    ap.add_argument("--max-chars", type=int, default=90_000,
                    help="пропускать длиннее — не влезает в окно судьи")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv.field_size_limit(2**31 - 1)
    with open(args.index, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"строк в индексе: {len(rows)}")

    pool = [r for r in rows
            if r["label"] in args.classes and int(r["prompt_chars"]) <= args.max_chars]
    print(f"после фильтра по классам и длине ({args.max_chars} симв.): {len(pool)}")

    by: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in pool:
        by[(stratum(r), r["label"])].append(r)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    short: list[str] = []
    for key in sorted(by, key=lambda k: (str(k[0]), k[1])):
        group = by[key]
        rng.shuffle(group)
        take = group[: args.per_stratum]
        picked.extend(take)
        if len(take) < args.per_stratum:
            short.append(f"{key[0]} класс {key[1]}: {len(take)} из {args.per_stratum}")

    print(f"отобрано {len(picked)} строк из {len(by)} страт")
    if short:
        print("страты, где не набралось:")
        for s in short[:12]:
            print(f"  {s}")
        if len(short) > 12:
            print(f"  ... и ещё {len(short)-12}")

    # Материализация: один проход по каждому файлу.
    want: dict[str, dict[tuple[int, int], dict]] = collections.defaultdict(dict)
    for r in picked:
        want[r["source_file"]][(int(r["sim_index"]), int(r["turn_idx"]))] = r

    data_dir = Path(args.data).expanduser()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["id", "prompt", "response", "label", "label_source", "gates",
                        "domain", "turn_idx", "has_tool_calls", "reward", "source_file"],
        )
        w.writeheader()
        for i, (src, needed) in enumerate(sorted(want.items()), 1):
            path = data_dir / src
            if not path.exists():
                print(f"  НЕТ ФАЙЛА {src[:60]}")
                continue
            for tt in iter_turns(path, only_reward=None, require_tool_calls=False):
                key = (tt.sim_index, tt.turn_idx)
                r = needed.pop(key, None)
                if r is None:
                    continue
                prompt, response = render_example(tt)
                w.writerow({
                    "id": r["id"], "prompt": prompt, "response": response,
                    "label": r["label"], "label_source": r["label_source"],
                    "gates": r["gates"], "domain": r["domain"],
                    "turn_idx": r["turn_idx"], "has_tool_calls": r["has_tool_calls"],
                    "reward": r["reward"], "source_file": src,
                })
                done += 1
                if not needed:
                    break
            print(f"  [{i}/{len(want)}] {src[:56]:56s} материализовано {done}", flush=True)

    print(f"\nзаписано {done} строк → {out}")
    c = collections.Counter((r["label"], r["domain"]) for r in picked)
    print("состав отобранного:")
    for k in sorted(c):
        print(f"  метка {k[0]:>2s}  {k[1]:20s} {c[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

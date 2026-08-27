"""Воспроизводит числа из `docs/data-findings.md`.

Существует, чтобы цифры в документах можно было перепроверить одной командой, а не верить
им на слово. Если организаторы выложат новые данные — запустить на них и сверить, что
выводы не поменялись.

    python scripts/probe_data.py --data data/raw/valid.parquet
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from sverka.data import load_examples
from sverka.transcript import parse_response, parse_transcript

LB, RB = "⟦", "⟧"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/valid.parquet")
    args = ap.parse_args()

    ex = load_examples(args.data)
    print(f"строк: {len(ex)}   диалогов: {len({e.conv for e in ex})}")

    labels = [e.label for e in ex if e.label is not None]
    if labels:
        print(f"баланс: {labels.count(1)} положительных / {labels.count(0)} отрицательных")
        pos_expl = sum(1 for e in ex if e.label == 1 and e.explanation)
        neg_expl = sum(1 for e in ex if e.label == 0 and e.explanation)
        print(f"explanation: у положительных {pos_expl}, у отрицательных {neg_expl}")

    print("\nдомены:", dict(Counter(e.domain for e in ex).most_common()))

    def q(vals, p):
        s = sorted(vals)
        return s[min(len(s) - 1, int(p * len(s)))]

    for field in ("prompt", "response"):
        lens = [len(getattr(e, field)) for e in ex]
        print(
            f"\n{field}: min={min(lens)} p50={q(lens,.5)} p90={q(lens,.9)} max={max(lens)} "
            f"сумма={sum(lens):,}"
        )
    ratio = sum(len(e.prompt) for e in ex) / max(1, sum(len(e.response) for e in ex))
    print(f"\nасимметрия контекст/ответ: {ratio:.0f}:1")

    markers = Counter()
    for e in ex:
        for m in re.findall(rf"{LB}([^{RB}]{{0,80}}){RB}", e.prompt):
            markers[m.split("·")[0].strip()] += 1
    print("\nмаркеры в prompt:", dict(markers.most_common()))

    ok_policy = ok_tools = ok_parse = 0
    tool_names: Counter = Counter()
    for e in ex:
        try:
            t = parse_transcript(e.prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"  НЕ РАЗОБРАЛСЯ {e.id}: {exc}")
            continue
        ok_parse += 1
        ok_policy += bool(t.policy)
        ok_tools += bool(t.tools)
        tool_names.update(t.tools.keys())

    print(f"\nразобрано: {ok_parse}/{len(ex)}   с политикой: {ok_policy}   со схемами: {ok_tools}")
    print(f"уникальных инструментов в схемах: {len(tool_names)}")
    for n, c in tool_names.most_common(15):
        print(f"   {n:35s} встречается в {c} примерах")

    calls = Counter()
    with_calls = 0
    for e in ex:
        turn = parse_response(e.response, strict=False)
        if turn.tool_calls:
            with_calls += 1
            calls.update(c.name for c in turn.tool_calls)
    print(f"\nответов с вызовами инструментов: {with_calls}/{len(ex)}")
    for n, c in calls.most_common(10):
        print(f"   {n:35s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

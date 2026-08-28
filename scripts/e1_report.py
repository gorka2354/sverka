"""E1, шаг 3 — разбор результатов прогона стокового ревьюера.

Считает не одно число, а пять, потому что одно здесь врёт.

**Согласие целиком** говорит мало: на выданном сэмпле баланс 50/50, но внутри страт он
совсем другой (0.91 против 0.09), и высокое согласие можно получить, просто угадывая
наличие вызова инструмента.

**Контроль по гейтам** отделяет «рубрика другая» от «модель не тянет». На строках, где
детерминированный гейт уже сработал, ответ известен без всякой модели. Если ревьюер
не находит ошибку даже там, узкое место — модель, и низкое согласие НЕ опровергает
гипотезу о происхождении меток.

**Формат-комплаенс** — доля ответов, из которых вообще удалось достать JSON. Прогон, где
треть ответов невалидна, измеряет форматирование, а не рубрику.

**Обрезанный контекст** отмечается отдельно: при превышении окна ollama срезает начало,
то есть политику, и такой вердикт бессмысленен по построению.

**κ Коэна** вместо accuracy: на несбалансированных подвыборках доля совпадений
систематически льстит.

    python scripts/e1_report.py --results e1_results.jsonl
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from pathlib import Path

# Вывод принудительно в UTF-8: на Windows консоль по умолчанию однобайтовая и роняет
# скрипт на первом же нелатинском символе — отчёт терялся целиком из-за оформления.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def kappa(y: list[int], p: list[int]) -> float:
    """κ Коэна. 0 — согласие на уровне случайного, 1 — полное."""
    n = len(y)
    if n == 0:
        return float("nan")
    obs = sum(1 for a, b in zip(y, p) if a == b) / n
    py1 = sum(y) / n
    pp1 = sum(p) / n
    exp = py1 * pp1 + (1 - py1) * (1 - pp1)
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan")


def prf(y: list[int], p: list[int]) -> tuple[float, float, float, int, int, int, int]:
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, p) if a == 0 and b == 0)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return f1, pr, rc, tp, fp, fn, tn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    raw = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    # Дедупликация: перезапуск прогона мог записать один пример дважды. Берём последнюю
    # запись — она сделана уже исправленным раннером.
    by_id: dict[str, dict] = {}
    for r in raw:
        by_id[r["id"]] = r
    rows = list(by_id.values())
    if len(rows) != len(raw):
        print(f"дублей убрано: {len(raw) - len(rows)}")
    print(f"строк в результатах: {len(rows)}")

    ok = [r for r in rows if r.get("json_ok")]
    bad = [r for r in rows if not r.get("json_ok")]
    trunc = [r for r in ok if r.get("truncated_ctx")]
    print(f"формат-комплаенс: {len(ok)}/{len(rows)} = {len(ok)/max(1,len(rows)):.2f}")
    if bad:
        empty = sum(1 for r in bad if (r.get("raw_len") or 0) == 0)
        print(f"  из невалидных пустых ответов: {empty}, с ошибкой: {sum(1 for r in bad if r.get('error'))}")
        print("  по размеру окна:", dict(collections.Counter(r.get("n_ctx") for r in bad)))
    if trunc:
        print(f"⚠️ с обрезанным контекстом: {len(trunc)} — у них срезано начало, то есть политика")

    usable = [r for r in ok if not r.get("truncated_ctx")]
    if not usable:
        print("годных строк нет")
        return 1

    y = [int(r["label"]) for r in usable]
    p = [int(r["pred_at_reviewed_turn"]) for r in usable]
    f1, pr, rc, tp, fp, fn, tn = prf(y, p)
    agree = sum(1 for a, b in zip(y, p) if a == b) / len(y)
    print()
    print(f"=== ГОДНЫХ СТРОК: {len(usable)} (доля позитива {sum(y)/len(y):.2f}) ===")
    print(f"согласие: {agree:.3f}   κ Коэна: {kappa(y,p):.3f}")
    print(f"F1={f1:.3f} P={pr:.3f} R={rc:.3f}  tp={tp} fp={fp} fn={fn} tn={tn}")

    print()
    print("=== КОНТРОЛЬ ИНТЕРПРЕТИРУЕМОСТИ ===")
    g = [r for r in usable if r.get("gate_fired")]
    if g:
        found = sum(1 for r in g if r["pred_at_reviewed_turn"] == 1)
        print(f"строк, где гейт уже дал ответ: {len(g)}; ревьюер нашёл ошибку в {found}")
        print(f"  → recall на заведомо разрешимых: {found/len(g):.2f}")
        print("  Если он заметно ниже единицы, узкое место — модель, и низкое общее")
        print("  согласие НЕ опровергает гипотезу о происхождении меток.")
    else:
        print("строк с сработавшим гейтом среди годных нет")

    print()
    print("=== ПО ДОМЕНАМ ===")
    for d in sorted({r["domain"] for r in usable}):
        idx = [i for i, r in enumerate(usable) if r["domain"] == d]
        f, pp, r_, *_ = prf([y[i] for i in idx], [p[i] for i in idx])
        print(f"  {d:20s} n={len(idx):2d} F1={f:.3f} P={pp:.3f} R={r_:.3f}")

    print()
    print("=== ЧТО РЕВЬЮЕР НАЗЫВАЕТ ПРИЧИНОЙ ===")
    tags = collections.Counter(t for r in usable if r["pred_at_reviewed_turn"] == 1 for t in (r.get("tags") or []))
    for t, n in tags.most_common(10):
        print(f"  {t:28s} {n}")

    print()
    print("=== ВЫВОД ПО ГЕЙТУ ЭКСПЕРИМЕНТА ===")
    if agree >= 0.9:
        print("согласие ≥ 0.9 → функция метки воспроизводится, корпус можно размечать прогоном")
    elif agree <= 0.6:
        print("согласие ≤ 0.6 → метки кастомные ЛИБО модель не тянет; смотреть контроль выше")
    else:
        print("согласие между 0.6 и 0.9 → однозначного вывода нет, решает контроль по гейтам")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

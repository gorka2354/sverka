"""Метрики с доверительным интервалом.

Ранжирование в соревновании идёт по Response F1. На 46 выданных примерах разница между
двумя решениями почти всегда лежит внутри доверительного интервала — поэтому интервал
считается всегда и печатается рядом с точечной оценкой. Это не украшение: без него легко
принять шум за прирост и потратить на него один из десяти сабмитов.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Scores:
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    n: int

    def __str__(self) -> str:
        return (
            f"F1={self.f1:.3f} P={self.precision:.3f} R={self.recall:.3f} "
            f"(tp={self.tp} fp={self.fp} fn={self.fn} tn={self.tn}, n={self.n})"
        )


def score(y_true: list[int], y_pred: list[int]) -> Scores:
    if len(y_true) != len(y_pred):
        raise ValueError(f"длины не совпадают: {len(y_true)} против {len(y_pred)}")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Scores(f1, precision, recall, tp, fp, fn, tn, len(y_true))


def bootstrap_f1(
    y_true: list[int], y_pred: list[int], *, n_iter: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Перцентильный бутстрэп-интервал для F1.

    Ресэмплинг идёт по примерам. Если валидация групповая (по диалогам), для честного
    интервала ресэмплить надо диалоги целиком — см. `bootstrap_f1_grouped`.
    """
    if not y_true:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(score([y_true[i] for i in idx], [y_pred[i] for i in idx]).f1)
    vals.sort()
    lo = vals[int((alpha / 2) * n_iter)]
    hi = vals[min(n_iter - 1, int((1 - alpha / 2) * n_iter))]
    return (lo, hi)


def bootstrap_f1_grouped(
    y_true: list[int],
    y_pred: list[int],
    groups: list[str],
    *,
    n_iter: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Бутстрэп с ресэмплингом ЦЕЛЫХ диалогов.

    Ходы одного диалога не независимы, и обычный бутстрэп по строкам даёт слишком узкий
    интервал — то есть врёт в опасную сторону, создавая иллюзию, что различия значимы.
    """
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    keys = sorted(by_group)
    if not keys:
        return (0.0, 0.0)

    rng = random.Random(seed)
    vals = []
    for _ in range(n_iter):
        idx: list[int] = []
        for _ in range(len(keys)):
            idx.extend(by_group[keys[rng.randrange(len(keys))]])
        vals.append(score([y_true[i] for i in idx], [y_pred[i] for i in idx]).f1)
    vals.sort()
    lo = vals[int((alpha / 2) * n_iter)]
    hi = vals[min(n_iter - 1, int((1 - alpha / 2) * n_iter))]
    return (lo, hi)

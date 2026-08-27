"""Загрузка датасета и сплит по диалогам.

Платформа передаёт тест в CSV (`--input <path_to_test.csv>`), а выданный сэмпл — parquet.
Поддерживаем оба, чтобы локальные прогоны и боевой шли через один код.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("id", "prompt", "response")


class DataError(RuntimeError):
    """Ошибка загрузки или формы датасета."""


@dataclass(frozen=True)
class Example:
    """Один пример: контекст, проверяемый ответ и (для размеченных) метка."""

    id: str
    prompt: str
    response: str
    label: int | None = None
    explanation: str | None = None

    @property
    def conv(self) -> str:
        """Идентификатор диалога — часть id до `::`.

        Ходы одного диалога коррелируют (общая политика, общий пользователь, общий контекст),
        поэтому сплит делается по этому полю, а не по строкам.
        """
        return self.id.split("::")[0]

    @property
    def domain(self) -> str:
        """Домен задачи: airline / retail / telecom / banking_knowledge."""
        return self.conv.split("__")[0]

    @property
    def turn(self) -> int | None:
        """Номер хода из id (`...::t19` → 19), None если не закодирован."""
        tail = self.id.split("::")[-1]
        if tail.startswith("t") and tail[1:].isdigit():
            return int(tail[1:])
        return None


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Прочитать parquet или CSV, проверив обязательные колонки."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"файл не найден: {p}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    else:
        raise DataError(f"неизвестный формат: {p.suffix} (ожидались .parquet/.csv)")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"нет обязательных колонок {missing}; есть: {list(df.columns)}")
    return df


def load_examples(path: str | Path) -> list[Example]:
    """Датасет → список Example. Метка и объяснение опциональны (в тесте их нет)."""
    df = load_dataframe(path)
    has_label = "label" in df.columns
    has_expl = "explanation" in df.columns

    out: list[Example] = []
    for row in df.to_dict("records"):
        expl = row.get("explanation") if has_expl else None
        # explanation заполнен только у положительных примеров; у остальных приходит NaN
        if expl is not None and (pd.isna(expl) or str(expl).strip().lower() in ("", "nan")):
            expl = None
        label = int(row["label"]) if has_label and not pd.isna(row["label"]) else None
        out.append(
            Example(
                id=str(row["id"]),
                prompt=str(row["prompt"]),
                response=str(row["response"]),
                label=label,
                explanation=str(expl) if expl is not None else None,
            )
        )
    return out


def split_by_conversation(
    examples: list[Example], *, n_folds: int = 5, seed: int = 0
) -> list[list[Example]]:
    """Разбить на фолды ПО ДИАЛОГАМ, стратифицируя по (домен, метка диалога).

    Сплит по строкам дал бы утечку: ходы одного разговора делят политику и историю,
    и модель, увидевшая один ход, «узнаёт» остальные. Метка диалога для стратификации —
    максимум по его ходам (диалог считается «грязным», если в нём есть хоть одна галлюцинация).
    """
    if n_folds < 2:
        raise DataError("n_folds должен быть >= 2")

    by_conv: dict[str, list[Example]] = {}
    for ex in examples:
        by_conv.setdefault(ex.conv, []).append(ex)

    def strata(conv_examples: list[Example]) -> tuple[str, int]:
        labels = [e.label for e in conv_examples if e.label is not None]
        return (conv_examples[0].domain, max(labels) if labels else -1)

    convs = sorted(by_conv)
    groups: dict[tuple[str, int], list[str]] = {}
    for c in convs:
        groups.setdefault(strata(by_conv[c]), []).append(c)

    import random

    folds: list[list[Example]] = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for _, conv_list in sorted(groups.items()):
        shuffled = list(conv_list)
        rng.shuffle(shuffled)
        # раскладываем диалоги страты по фолдам по кругу — так каждая страта
        # представлена во всех фолдах примерно поровну
        for i, c in enumerate(shuffled):
            folds[i % n_folds].extend(by_conv[c])
    return folds

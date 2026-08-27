"""Порт детектора и тривиальные бейзлайны.

Каскад из inc-3 будет списком объектов этого интерфейса, поэтому вклад каждого слоя
измеряется отдельно, а не только итог. Бейзлайн — такой же детектор: сравнение с ним
идёт через тот же стенд, без отдельного кода.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from poverka.data import Example


@dataclass(frozen=True)
class Verdict:
    """Решение детектора по одному примеру.

    `evidence` — обязательное поле, а не украшение: без привязки к куску контекста нельзя
    отличить «сработало правильно» от «совпало случайно», а на выборке в 46 примеров
    случайных совпадений будет много.
    """

    label: int
    confidence: float = 1.0
    reason: str = ""
    evidence: list[str] = field(default_factory=list)


class Detector(Protocol):
    """Интерфейс детектора."""

    name: str

    def predict(self, example: Example) -> Verdict: ...


@dataclass
class ConstantDetector:
    """Всегда предсказывает одну метку. Пол, ниже которого решение бессмысленно.

    При балансе 50/50 константа «всё 1» даёт F1 = 0.667 — это и есть планка,
    которую обязана превзойти любая содержательная модель.
    """

    value: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"constant{self.value}"

    def predict(self, example: Example) -> Verdict:
        return Verdict(label=self.value, confidence=0.5, reason=f"константа {self.value}")


@dataclass
class RandomDetector:
    """Случайное предсказание. Нужен, чтобы видеть уровень шума на этой выборке."""

    p: float = 0.5
    seed: int = 0
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def predict(self, example: Example) -> Verdict:
        v = 1 if self._rng.random() < self.p else 0
        return Verdict(label=v, confidence=0.5, reason="случайно")


@dataclass
class ResponseLengthDetector:
    """Проверка на глупую корреляцию: длинный ответ = галлюцинация.

    Существует не как кандидат в решение, а как контроль. Если такой детектор даёт заметный
    F1, значит в выборке есть артефакт длины, и на него легко переобучиться, приняв его
    за содержательный сигнал.
    """

    threshold: int = 427  # медиана длины response в выданном сэмпле
    name: str = "response_length"

    def predict(self, example: Example) -> Verdict:
        n = len(example.response)
        return Verdict(
            label=1 if n > self.threshold else 0,
            confidence=0.5,
            reason=f"len(response)={n} против порога {self.threshold}",
        )


def run_detector(detector: Detector, examples: list[Example]) -> list[Verdict]:
    return [detector.predict(ex) for ex in examples]

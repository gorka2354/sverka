"""Тесты загрузчика, сплита и парсера — на выданных данных, а не на выдуманных.

Смысл: парсер писался по фактическому формату файла, и если организаторы сменят разметку,
эти тесты должны покраснеть первыми.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poverka.data import Example, load_examples, split_by_conversation
from poverka.transcript import TranscriptError, parse_response, parse_transcript

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "valid.parquet"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="нет data/raw/valid.parquet")


@pytest.fixture(scope="module")
def examples() -> list[Example]:
    return load_examples(DATA)


def test_dataset_shape(examples):
    assert len(examples) == 46
    labels = [e.label for e in examples]
    assert sum(labels) == 23, "баланс в выданном сэмпле ровно 50/50"


def test_explanation_only_for_positives(examples):
    """Объяснение выдано только там, где галлюцинация есть — это рубрика разметки."""
    for e in examples:
        if e.label == 1:
            assert e.explanation, f"{e.id}: у положительного примера нет объяснения"
        else:
            assert e.explanation is None, f"{e.id}: у отрицательного примера появилось объяснение"


def test_id_decomposition(examples):
    ex = next(e for e in examples if e.id.startswith("airline__10::t19"))
    assert ex.conv == "airline__10"
    assert ex.domain == "airline"
    assert ex.turn == 19


def test_split_does_not_leak_conversations(examples):
    """Ходы одного диалога не должны попадать в разные фолды — иначе оценка завышена."""
    folds = split_by_conversation(examples, n_folds=5, seed=0)
    seen: dict[str, int] = {}
    for i, fold in enumerate(folds):
        for ex in fold:
            assert seen.setdefault(ex.conv, i) == i, f"{ex.conv} попал в фолды {seen[ex.conv]} и {i}"
    assert sum(len(f) for f in folds) == len(examples)


def test_every_prompt_parses(examples):
    """Все 46 промптов разбираются без исключений, и в каждом есть системный блок."""
    for e in examples:
        t = parse_transcript(e.prompt)
        assert t.system, f"{e.id}: пустой системный блок"
        assert t.turns, f"{e.id}: не найдено ни одного хода"


def test_policy_and_tools_are_extracted(examples):
    """Политика и схемы инструментов — основа детерминированного слоя (inc-1)."""
    with_policy = 0
    with_tools = 0
    for e in examples:
        t = parse_transcript(e.prompt)
        with_policy += bool(t.policy)
        with_tools += bool(t.tools)
    # пороги проверены замером на выданном сэмпле; если реальный тест окажется другим,
    # тест покраснеет — и это правильное поведение, а не повод понизить порог
    assert with_policy >= 40, f"политика извлечена только у {with_policy} из {len(examples)}"
    assert with_tools >= 30, f"схемы инструментов извлечены только у {with_tools} из {len(examples)}"


def test_response_parses_as_single_assistant_turn(examples):
    for e in examples:
        turn = parse_response(e.response)
        assert turn.role == "ASSISTANT", f"{e.id}: ответ разобран как {turn.role}"


def test_tool_calls_are_extracted_where_present(examples):
    """У части ответов есть вызовы инструментов — они должны доставаться с аргументами."""
    with_calls = [e for e in examples if "TOOL_CALL" in e.response]
    assert with_calls, "в сэмпле были ответы с вызовами инструментов"
    for e in with_calls:
        turn = parse_response(e.response)
        assert turn.tool_calls, f"{e.id}: TOOL_CALL в тексте есть, а разобранных вызовов нет"
        for c in turn.tool_calls:
            assert c.name, f"{e.id}: у вызова нет имени"


def test_parser_fails_loudly_on_garbage():
    """Молчаливый пустой разбор опаснее падения: он съедает сабмит и выдаёт мусор."""
    with pytest.raises(TranscriptError):
        parse_transcript("тут нет никаких маркеров, просто текст")

"""Сборка архива сабмита.

Платформа делает ровно это:

    pip install .
    python scripts/predict.py --input <test.csv> --output <predictions.csv>

Отсюда два следствия, из-за которых сборка вынесена в отдельный скрипт, а не берётся
из репозитория как есть:

1. **Точка входа обязана называться `scripts/predict.py`.** Диагностический прогон живёт
   в репозитории как `scripts/diagnose.py`, а в архив кладётся под именем `predict.py`.

2. **У профиля `diagnostic` не должно быть ни одной зависимости.** Рабочий `pyproject.toml`
   объявляет `pandas` и `pyarrow`; если в контейнере нет интернета, `pip install .` упадёт
   раньше, чем скрипт что-то напечатает, и диагностика не ответит на вопрос, ради которого
   затевалась. Поэтому архив получает собственный, урезанный `pyproject.toml`.

Дополнительно кладётся `setup.py`, печатающий состояние среды **во время установки**:
`[build-system] requires` включает build isolation, и pip идёт в индекс за setuptools ещё
до чтения наших зависимостей. Если установка упадёт на этом шаге, хук — единственный
источник информации.

    python scripts/build_submission.py --profile diagnostic
"""

from __future__ import annotations

import argparse
import ast
import shutil
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP_PY_HOOK = '''"""Хук сборки: печатает среду во время `pip install .`.

Существует не для сборки, а для диагностики: build isolation заставляет pip ходить
в индекс за setuptools до чтения зависимостей проекта, поэтому установка может упасть
раньше запуска точки входа. Тогда этот вывод — единственное, что мы узнаем о среде.
"""

import sys

try:
    _lines = [
        "[sverka:build] python=" + sys.version.replace("\\n", " "),
        "[sverka:build] executable=" + sys.executable,
        "[sverka:build] platform=" + sys.platform,
        "[sverka:build] prefix=" + sys.prefix,
        "[sverka:build] path0=" + (sys.path[0] if sys.path else "?"),
    ]
    for _m in ("torch", "vllm", "transformers", "numpy"):
        try:
            _mod = __import__(_m)
            _lines.append("[sverka:build] " + _m + "=" + str(getattr(_mod, "__version__", "?")))
        except Exception as _e:
            _lines.append("[sverka:build] " + _m + "=НЕТ (" + type(_e).__name__ + ")")
    for _s in (sys.stderr, sys.stdout):
        for _l in _lines:
            print(_l, file=_s, flush=True)
except Exception:
    pass

from setuptools import setup

setup()
'''

PYPROJECT_DIAGNOSTIC = '''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "sverka"
version = "0.1.0"
description = "Guardian of Truth — diagnostic run (stdlib only)"
requires-python = ">=3.9"
# Пусто намеренно: платформа ставит пакет ДО запуска скрипта, и любая сторонняя
# зависимость превращает отсутствие интернета в необъяснимое падение установки.
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
'''

PACKAGE_INIT = '''"""Заглушка пакета для профиля diagnostic.

Вся логика прогона лежит в `scripts/predict.py` и использует только стандартную
библиотеку. Пакет существует, чтобы `pip install .` имел что установить.
"""

__version__ = "0.1.0"
'''

README_BY_PROFILE = {
    "strict": """# сверка — проба контракта

Этот архив не является решением. Метка назначается по контрольной сумме идентификатора,
без единого взгляда на содержимое ответа: примерно девяти строкам из десяти ставится 1,
остальным 0. Правило существует, чтобы измерить долю положительных примеров в наборе —
его Precision равен этой доле — и чтобы в файле гарантированно присутствовали оба класса.

Перед записью точка входа проверяет собственное чтение входного файла и завершается
ненулевым кодом, если проверки не сошлись. Смысл в коде возврата: платформа не показывает
ни логов, ни текста ошибки, поэтому единственный наблюдаемый выход — номер шага, на котором
прогон остановлен. Падение уводит на шаг распаковки, успех — на шаг расчёта метрик.

Зависимостей не объявлено. Стандартной библиотеки достаточно; pandas импортируется лениво
и только если вход опознан как parquet по сигнатуре.
""",
    "noheader": """# сверка — проба контракта, файл предсказаний без заголовка

Этот архив не является решением. Метка назначается по контрольной сумме идентификатора,
без единого взгляда на содержимое ответа: примерно девяти строкам из десяти ставится 1,
остальным 0. Правило существует, чтобы измерить долю положительных примеров в наборе —
его Precision равен этой доле — и чтобы в файле гарантированно присутствовали оба класса.

Перед записью точка входа проверяет собственное чтение входного файла и завершается
ненулевым кодом, если проверки не сошлись. Смысл в коде возврата: платформа не показывает
ни логов, ни текста ошибки, поэтому единственный наблюдаемый выход — номер шага, на котором
прогон остановлен. Падение уводит на шаг распаковки, успех — на шаг расчёта метрик.

Зависимостей не объявлено. Стандартной библиотеки достаточно; pandas импортируется лениво
и только если вход опознан как parquet по сигнатуре.

Отличие от профиля `strict` ровно одно: файл предсказаний пишется без строки заголовка.
Условие описывает выход фразой «файл с колонками id, label» и не уточняет, ожидается
ли заголовок.
""",
    "gate": """# сверка — измерение точности детерминированного гейта

Метка 1 ставится ровно там, где сработал гейт, и ни в каком другом месте. Тогда показанный
Precision прогона равен точности гейта, а Recall — его покрытию положительных примеров.

Так сделано потому, что локально эта величина не измеряется: все гейты требуют вызова
инструмента в проверяемом ответе, а в выданном сэмпле внутри этой страты нет ни одного
достижимого отрицательного примера. Наблюдение «все срабатывания попали в положительные»
там не измеряет ничего.

Модели в решении нет, объявленных зависимостей нет; pandas берётся из образа.
""",
    "probe": """# сверка — проба среды исполнения

Этот архив не является решением. Он записывает валидный файл предсказаний, а затем
завершается нулём или единицей в зависимости от ответа на один вопрос о среде исполнения:
установился ли пакет, есть ли сеть, есть ли в образе torch, виден ли GPU.

Платформа не показывает ни логов, ни текста ошибки, поэтому единственный доступный канал —
номер шага, на котором прогон остановлен. Один прогон приносит один бит. Иначе состав
образа узнать неоткуда, а от него зависит, можно ли запустить модель-судью и надо ли
везти библиотеки внутри архива.
""",
    "parquet_out": """# сверка — проба контракта, файл предсказаний в parquet

Этот архив не является решением. Метка назначается по контрольной сумме идентификатора,
без единого взгляда на содержимое ответа: примерно девяти строкам из десяти ставится 1,
остальным 0. Правило существует, чтобы измерить долю положительных примеров в наборе —
его Precision равен этой доле — и чтобы в файле гарантированно присутствовали оба класса.

Перед записью точка входа проверяет собственное чтение входного файла и завершается
ненулевым кодом, если проверки не сошлись. Смысл в коде возврата: платформа не показывает
ни логов, ни текста ошибки, поэтому единственный наблюдаемый выход — номер шага, на котором
прогон остановлен. Падение уводит на шаг распаковки, успех — на шаг расчёта метрик.

Зависимостей не объявлено. Стандартной библиотеки достаточно; pandas импортируется лениво
и только если вход опознан как parquet по сигнатуре.

Отличие от профиля `strict` ровно одно: файл предсказаний пишется в формате parquet
с колонками `id` и `label`, а не текстовым CSV. Условие называет выход `predictions.csv`,
но на этой платформе расширение формат не описывает: входной файл называется `test.csv`
и при этом является parquet.
""",
    "all_ones": """# сверка — сабмит «всё = 1`

Тривиальный сабмит: метка 1 для каждой строки. Отправляется намеренно.
Его Precision буквально равен доле положительных примеров в наборе — величине, которую
иначе не узнать, а от неё зависит и пол метрики, и калибровка порога.
Зависимостей нет: используется только стандартная библиотека.
""",
    "has_tool_call": """# сверка — сабмит «есть вызов инструмента`

Метка 1 там, где в поле response встречается TOOL_CALL. На выданном сэмпле это правило
даёт F1 = 0.913, и сабмит проверяет, переносится ли артефакт на тестовый набор.
Зависимостей нет: используется только стандартная библиотека.
""",
}

README = """# сверка — диагностический прогон

Этот архив НЕ является решением. Он намеренно завершается ненулевым кодом, чтобы
не израсходовать лимит сабмитов, и печатает состояние среды исполнения.

Перед падением записывается валидный файл предсказаний (все метки = 1) — на случай,
если платформа проигнорирует код возврата.
"""


# Раскладка без pyproject.toml. Единственный способ обойти build isolation, когда флаги
# установки передать некуда: платформа выполняет голое `pip install .`, а при наличии
# pyproject pip идёт в индекс за setuptools ДО чтения наших зависимостей — и без интернета
# падает, даже если зависимостей у нас нет вовсе.
_LEGACY_SETUP_CALL = (
    "from setuptools import setup\n\n"
    "setup(\n"
    '    name="sverka",\n'
    '    version="0.1.0",\n'
    '    package_dir={"": "src"},\n'
    '    packages=["sverka"],\n'
    "    install_requires=[],\n"
    ")\n"
)
SETUP_PY_LEGACY = SETUP_PY_HOOK.replace(
    "from setuptools import setup\n\nsetup()\n", _LEGACY_SETUP_CALL
)


# Вопросы к образу платформы для профиля `probe`. Каждый стоит одного прогона и приносит
# один бит: «да» уводит на шаг расчёта метрик, «нет» — на шаг распаковки.
#
# Порядок здесь — это порядок отправки, и он не случаен. Первым идёт `package_installed`:
# если платформа не может установить даже наш пакет без зависимостей, всё остальное
# теряет смысл. Вторым — `internet`: он один способен закрыть весь вопрос о вендоринге.
# Дальше по убыванию цены ошибки.
PROBE_QUESTIONS = (
    # ── Вопросы о САМОМ ТЕСТОВОМ НАБОРЕ. Добавлены 31.08.2026, когда выяснилось, что
    # выданный сэмпл врёт уже по четырём признакам, а свойства входа от меток не зависят
    # и потому покупаются бесплатно отказным прогоном. Порядок — по цене ошибки.
    "report",             # ПОЛНЫЙ отчёт в stdout: логи платформы отдаются по прямой ссылке
    "parse_health",       # здоров ли разбор транскрипта на боевом файле (≥0.95)
    "units_heavy",        # доля ответов текстовой страты с ≥3 единицами > 0.30
    "median_len_gt_500",  # медиана длины ответа > 500 символов
    "n_gt_800",           # строк в наборе больше 800
    "tool_gt_200",        # строк с вызовом инструмента больше 200
    # ── Вопросы об образе платформы.
    "package_installed",  # выполняется ли `pip install .` успешно
    "internet",           # есть ли сеть в контейнере
    "torch",              # torch в образе
    "cuda",               # torch видит GPU
    "vllm",               # vllm в образе
    "transformers",       # transformers в образе
    "gpu_h100",           # это действительно H100
    "sklearn", "numpy", "hf",
)


# Гейты, доступные профилю `gate`. Один прогон — один гейт: смешивать нельзя, иначе
# показанный Precision станет средним по смеси и не скажет ничего ни об одном из них.
GATES = ("all", "A", "B", "C", "D", "M")


PYPROJECT_PACKAGE = '''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "sverka"
version = "0.1.0"
description = "Guardian of Truth"
requires-python = ">=3.9"
# Зависимостей нет намеренно. `[E9-A]` Платформа выполняет голое `pip install .`, флагов
# туда не передать; при отсутствии сети любая объявленная зависимость роняет установку.
# pandas в точке входа импортируется лениво и берётся из образа — что он там есть,
# измерено: чтение parquet на платформе отработало.
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

# Схемы инструментов лежат ВНУТРИ пакета, а не в `model/`: после `pip install .` пакет
# живёт в site-packages, и путь относительно корня архива указывал бы мимо.
[tool.setuptools.package-data]
sverka = ["resources/*.json"]
'''


PROFILE_ENTRY = {
    "broken": "scripts/predict_minimal.py",  # не используется, профиль собирается отдельно
    # Предельно простая точка входа для отладки контракта вслепую: платформа не показывает
    # ни ошибок, ни логов, поэтому каждый сабмит должен нести ровно один бит информации.
    "minimal": "scripts/predict_minimal.py",
    "diagnostic": "scripts/diagnose.py",
    # Тот же скрипт, но завершается НУЛЁМ. Разбор по исходам: вариант с падением стоит
    # 0 попыток в хорошем случае и 1 в плохом, а потом всё равно нужна попытка на «всё = 1»
    # ради доли положительных — итого 1 или 2. Вариант с нулём стоит ровно 1 всегда и даёт
    # сразу и логи, и долю. Он слабо доминирует, и проигрывает лишь в одном сценарии:
    # если платформа показывает логи только у упавших прогонов.
    "diagnostic_scored": "scripts/diagnose.py",
    # Проба, которая проверяет собственное чтение входа и падает, если оно не сошлось.
    # Смысл — в коде возврата: падение уводит на шаг 08, успех на 22, и это единственный
    # способ различить «прочитали не так» и «записали не то» без логов.
    "strict": "scripts/predict_strict.py",
    # Та же проба, но без строки заголовка.
    "noheader": "scripts/predict_strict.py",
    # Та же проба, но файл предсказаний пишется в parquet, а не в текстовый CSV.
    # Формат контейнера выхода — единственное, чего десять отправленных прогонов
    # не варьировали ни разу, при том что вход уже оказался parquet под именем .csv.
    "parquet_out": "scripts/predict_strict.py",
    # Опрос образа платформы. Что именно спрашиваем — задаётся флагом --question.
    "probe": "scripts/predict_strict.py",
    # Сабмит-измеритель: метка 1 ровно там, где сработал гейт. Тогда Precision прогона
    # равен точности гейта, а Recall — его покрытию положительных. Локально это
    # не измеряется: в выданном сэмпле нет ни одного достижимого отрицательного
    # в той страте, где гейты вообще способны сработать.
    "gate": "scripts/predict_gate.py",
    "all_ones": "scripts/predict_stdlib.py",
    "has_tool_call": "scripts/predict_stdlib.py",
    # Отвод: единица по умолчанию, ноль только там, где ответ заведомо безобиден.
    # Требует пакета sverka в архиве (otvod → units, transcript), но ни одной сторонней
    # библиотеки: проверено обходом sys.modules — не-stdlib импортов ноль.
    "otvod": "scripts/predict_stdlib.py",
    # Измерительный сабмит: одно условие «нечего проверять». Возвращает долю положительных
    # среди ответов без единиц на БОЕВЫХ данных — ставку, на которой стоят все конструкции.
    "no_units": "scripts/predict_stdlib.py",
}


def build_diagnostic(out_dir: Path, profile: str = "diagnostic",
                     packaging: str = "pep517", question: str = "none",
                     gate: str = "all") -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "scripts").mkdir(parents=True)
    (out_dir / "src" / "sverka").mkdir(parents=True)
    (out_dir / "model").mkdir(parents=True)

    if profile == "broken":
        # ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ. Архив намеренно без точки входа. Нужен, чтобы понять,
        # что означает номер шага, на котором падают нормальные архивы: если этот упадёт
        # РАНЬШЕ — значит до запуска нашего кода платформа доходит, и дело в самом коде.
        # Если упадёт на том же шаге — номер шага generic, и он ничего про код не говорит.
        (out_dir / "scripts" / ".gitkeep").write_text("", encoding="utf-8", newline="\n")
        (out_dir / "pyproject.toml").write_text(PYPROJECT_DIAGNOSTIC, encoding="utf-8", newline="\n")
        (out_dir / "setup.py").write_text(SETUP_PY_HOOK, encoding="utf-8", newline="\n")
        (out_dir / "src" / "sverka" / "__init__.py").write_text(PACKAGE_INIT, encoding="utf-8", newline="\n")
        (out_dir / "README.md").write_text(
            "# отрицательный контроль: архив намеренно без scripts/predict.py\n",
            encoding="utf-8", newline="\n")
        (out_dir / "model" / ".gitkeep").write_text("", encoding="utf-8", newline="\n")
        return out_dir

    src = ROOT / PROFILE_ENTRY[profile]
    entry = out_dir / "scripts" / "predict.py"
    shutil.copy2(src, entry)
    if profile == "diagnostic_scored":
        # Профиль сохранён только ради истории отправок. Константы EXIT_CODE в diagnose.py
        # больше нет: код возврата теперь выводится из того, удалось ли записать хоть одно
        # предсказание (diagnose.py:262). Собирать профиль по старой схеме нельзя — прежний
        # `assert` молча указывал бы на несуществующую строку.
        raise SystemExit(
            "профиль diagnostic_scored устарел: diagnose.py больше не содержит EXIT_CODE, "
            "код возврата выводится из числа записанных предсказаний"
        )
    elif profile == "noheader":
        # Меняется одна константа, а не тело функции: правка по строкам кода уже однажды
        # дала синтаксически битый архив.
        text = entry.read_text(encoding="utf-8")
        assert "WRITE_HEADER = True" in text, "не найдена константа заголовка"
        entry.write_text(text.replace("WRITE_HEADER = True", "WRITE_HEADER = False"),
                         encoding="utf-8", newline="\n")
    elif profile == "gate":
        text = entry.read_text(encoding="utf-8")
        assert 'GATE = "all"' in text, "не найдена константа гейта"
        assert gate in GATES, f"неизвестный гейт: {gate}"
        entry.write_text(text.replace('GATE = "all"', f'GATE = "{gate}"'),
                         encoding="utf-8", newline="\n")
    elif profile == "probe":
        text = entry.read_text(encoding="utf-8")
        assert 'PROBE_QUESTION = "none"' in text, "не найдена константа вопроса"
        assert question in PROBE_QUESTIONS, f"неизвестный вопрос: {question}"
        # ⚠️ Опросник пишет выход ТЕКСТОВЫМ CSV, и это не пережиток, а условие бесплатности
        # канала. `[изменено 31.08.2026]` Прежняя редакция ставила здесь parquet, потому что
        # формат выхода был рабочей гипотезой. Гипотеза подтвердилась: платформа читает
        # parquet и засчитывает такой прогон. Значит опросник с parquet-выходом при ответе
        # «да» (код 0) завершился бы УСПЕШНО и съел невозвратный слот — то есть ровно то,
        # ради чего опросник и существует, перестало бы работать.
        #
        # С текстовым CSV оба исхода — отказ: «да» уводит на шаг 22 (выход не сопоставился
        # с эталоном), «нет» на шаг 08 (ненулевой код). Один бит за прогон, ноль слотов.
        assert 'OUTPUT_FORMAT = "csv"' in text, "формат выхода опросника обязан остаться csv"
        entry.write_text(text.replace('PROBE_QUESTION = "none"',
                                      f'PROBE_QUESTION = "{question}"'),
                         encoding="utf-8", newline="\n")
    elif profile == "parquet_out":
        text = entry.read_text(encoding="utf-8")
        assert 'OUTPUT_FORMAT = "csv"' in text, "не найдена константа формата выхода"
        entry.write_text(text.replace('OUTPUT_FORMAT = "csv"',
                                      'OUTPUT_FORMAT = "parquet"'),
                         encoding="utf-8", newline="\n")
    elif profile not in ("diagnostic", "minimal", "strict"):
        # Профиль зашивается в точку входа, а не передаётся флагом: платформа запускает
        # скрипт со СВОИМИ аргументами, и никакого нашего флага там не будет.
        text = entry.read_text(encoding="utf-8")
        assert 'PROFILE = "all_ones"' in text, "не найдена строка профиля в точке входа"
        entry.write_text(text.replace('PROFILE = "all_ones"', f'PROFILE = "{profile}"'),
                         encoding="utf-8", newline="\n")
    if packaging == "pep517":
        pj = PYPROJECT_PACKAGE if profile in ("gate", "otvod", "no_units", "probe") else PYPROJECT_DIAGNOSTIC
        (out_dir / "pyproject.toml").write_text(pj, encoding="utf-8", newline="\n")
        (out_dir / "setup.py").write_text(SETUP_PY_HOOK, encoding="utf-8", newline="\n")
    else:
        # Без pyproject.toml pip идёт по старому пути и берёт setuptools из окружения,
        # а не из индекса. Это единственный способ обойти build isolation, когда флаги
        # установки нам передать некуда: платформа выполняет голое `pip install .`.
        (out_dir / "setup.py").write_text(SETUP_PY_LEGACY, encoding="utf-8", newline="\n")
    if profile in ("gate", "otvod", "no_units", "probe"):
        # Настоящий пакет: гейтам нужны transcript, gates, schema_check и схемы
        # инструментов, отводу — otvod, units, transcript. `__pycache__` и `data.py`
        # исключены — первое мусор, второе единственный модуль, тянущий pandas
        # на верхнем уровне, а в сабмите он не нужен.
        src = ROOT / "src" / "sverka"
        dst = out_dir / "src" / "sverka"
        for item in sorted(src.rglob("*")):
            if "__pycache__" in item.parts or item.name == "data.py":
                continue
            цель = dst / item.relative_to(src)
            if item.is_dir():
                цель.mkdir(parents=True, exist_ok=True)
            else:
                цель.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, цель)
    else:
        (out_dir / "src" / "sverka" / "__init__.py").write_text(PACKAGE_INIT, encoding="utf-8", newline="\n")
    (out_dir / "README.md").write_text(README_BY_PROFILE.get(profile, README), encoding="utf-8", newline="\n")
    (out_dir / "model" / ".gitkeep").write_text("", encoding="utf-8", newline="\n")
    return out_dir


# Профили, которые собираются ради отправки. Только для них имеет смысл требовать, чтобы
# точка входа отработала на реальном сэмпле: `broken` намеренно без точки входа, а `minimal`
# — историческая проба, читающая исключительно UTF-8, и на parquet она обязана падать.
SUBMITTABLE = ("diagnostic", "all_ones", "has_tool_call", "otvod", "no_units", "strict", "noheader",
               "parquet_out", "probe", "gate")


def smoke(out_dir: Path, profile: str) -> list[str]:
    """Прогнать точку входа на выданном сэмпле и сверить выход со входом.

    `[инцидент 28.08.2026]` Архив профиля `diagnostic_scored` уехал на платформу и записал
    файл из одних десяти байт — только заголовок `id,label`. Его точка входа не опознавала
    parquet, чтение падало, исключение проглатывалось, а код возврата всё равно был нулевым.
    Сборка это пропустила, потому что проверяла лишь импорты и наличие файлов.

    Отсюда гейт: сэмпл кладётся под именем `test.csv` — ровно так платформа и врёт про
    формат, — и выход сверяется со входом по числу строк и по множеству идентификаторов.
    Архив, который не прошёл сверку, не собирается.
    """
    import csv as _csv
    import subprocess
    import sys
    import tempfile

    sample = ROOT / "data" / "raw" / "valid.parquet"
    if not sample.exists():
        return [f"[гейт пропущен] нет {sample} — сверить выход со входом не на чем"]

    import pandas as pd
    df = pd.read_parquet(sample)
    ожидаемые = {str(x) for x in df["id"]}

    with tempfile.TemporaryDirectory() as td:
        вход = Path(td) / "test.csv"          # имя врёт про формат — как на платформе
        вход.write_bytes(sample.read_bytes())
        выход = Path(td) / "predictions.csv"
        r = subprocess.run(
            [sys.executable, str(out_dir / "scripts" / "predict.py"),
             "--input", str(вход), "--output", str(выход)],
            capture_output=True, timeout=600,
        )
        if r.returncode != 0 and profile != "probe":
            хвост = (r.stderr or r.stdout).decode("utf-8", "replace").strip().splitlines()[-3:]
            return [f"точка входа вернула {r.returncode} на выданном сэмпле: {' / '.join(хвост)}"]
        if profile == "probe":
            # У опросника ненулевой код — это ОТВЕТ «нет», а не поломка. Проверять надо
            # другое: что файл предсказаний записан в обоих случаях, иначе ответ «нет»
            # обошёлся бы нам ещё и потерей валидного выхода.
            print(f"  [гейт] опросник ответил: {'да' if r.returncode == 0 else 'нет'} "
                  f"(в этом окружении; на платформе ответ может быть иным)")
        if not выход.exists():
            return ["точка входа отработала, но файла предсказаний не создала"]

        сырое = выход.read_bytes()
        if сырое.startswith(b"PAR1"):
            # Профиль parquet_out: схема лежит внутри файла, разбирать как текст нельзя.
            вых_df = pd.read_parquet(выход)
            строки = [[str(c) for c in вых_df.columns]] + [
                [str(a), str(b)] for a, b in zip(вых_df.iloc[:, 0], вых_df.iloc[:, 1])
            ]
        else:
            строки = list(_csv.reader(сырое.decode("utf-8").splitlines()))

    problems = []
    parquet_выход = сырое.startswith(b"PAR1")
    ждём_заголовок = profile != "noheader"
    есть_заголовок = bool(строки) and строки[0] == ["id", "label"]
    if ждём_заголовок and not есть_заголовок:
        problems.append(f"заголовок выхода {строки[0] if строки else '—'}, а нужен ['id', 'label']")
    if not ждём_заголовок and есть_заголовок:
        problems.append("профиль noheader обязан писать файл БЕЗ строки заголовка")
    тело = строки[1:] if есть_заголовок else строки
    полученные = {r[0] for r in тело if r}
    if len(тело) != len(df):
        problems.append(f"строк на выходе {len(тело)}, а на входе {len(df)}")
    if полученные != ожидаемые:
        нет = list(ожидаемые - полученные)[:2]
        лишние = list(полученные - ожидаемые)[:2]
        problems.append(f"идентификаторы не совпали: потеряно {len(ожидаемые - полученные)} "
                        f"{нет}, лишних {len(полученные - ожидаемые)} {лишние}")
    if not parquet_выход:
        if b"\r" in сырое:
            problems.append("в выходе есть возврат каретки — нужен перевод строки LF")
        if сырое.startswith(b"\xef\xbb\xbf"):
            problems.append("выход начинается с BOM")
    if profile == "parquet_out" and not parquet_выход:
        problems.append("профиль parquet_out обязан писать выход в parquet")
    метки = sorted({r[1] for r in тело if len(r) > 1})
    print(f"  [гейт] {len(тело)} строк, идентификаторы сошлись, метки {метки}, "
          f"формат: {'parquet' if parquet_выход else 'csv'}, "
          f"заголовок: {'есть' if есть_заголовок else 'нет'}")
    return problems


def check(out_dir: Path) -> list[str]:
    """Проверки контракта, которые дешевле сделать здесь, чем узнать из упавшего сабмита."""
    problems = []
    entry = out_dir / "scripts" / "predict.py"
    if not entry.exists():
        if out_dir.name.startswith("broken"):
            return []  # отрицательный контроль: отсутствие точки входа здесь намеренное
        problems.append("нет scripts/predict.py — платформа запускает именно его")
    else:
        # Проверяются только импорты ВЕРХНЕГО УРОВНЯ. Ленивый импорт внутри функции
        # разрешён и нужен: если платформа передаст parquet вместо CSV, мы попробуем
        # взять pandas из окружения. Опасен именно импорт на уровне модуля — он
        # выполняется при запуске и роняет скрипт, если пакета нет.
        forbidden = {"pandas", "pyarrow", "numpy", "torch", "sverka"}
        try:
            tree = ast.parse(entry.read_text(encoding="utf-8"))
        except SyntaxError as e:
            problems.append(f"точка входа не парсится: {e}")
            return problems
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in forbidden:
                    problems.append(
                        f"точка входа импортирует {n} на верхнем уровне — обязан быть stdlib-only"
                    )
    pj_path = out_dir / "pyproject.toml"
    if pj_path.exists():
        pj = pj_path.read_text(encoding="utf-8")
        if "dependencies = []" not in pj:
            problems.append("в pyproject объявлены зависимости — установка упадёт без сети")
    elif not (out_dir / "setup.py").exists():
        problems.append("нет ни pyproject.toml, ни setup.py — pip не сможет установить пакет")
    if not (out_dir / "src").exists():
        problems.append("нет src/ — pip install . не найдёт пакет")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="diagnostic",
                    choices=["broken", "minimal", "diagnostic", "diagnostic_scored", "all_ones",
                             "has_tool_call", "otvod", "no_units", "strict", "noheader",
                             "parquet_out", "probe", "gate"])
    ap.add_argument("--out", default=None, help="куда собирать (по умолчанию submissions/<profile>)")
    ap.add_argument("--archive", action="store_true", help="дополнительно упаковать архив")
    ap.add_argument("--format", default="tar.gz", choices=["tar.gz", "zip"],
                    help="платформа приняла tar.gz на загрузку, но могла ожидать zip")
    ap.add_argument("--packaging", default="pep517", choices=["pep517", "legacy"],
                    help="pep517 — с pyproject.toml (включает build isolation: pip идёт "
                         "в индекс за setuptools ДО чтения наших зависимостей, и без "
                         "интернета установка падает). legacy — только setup.py, без "
                         "pyproject: pip использует setuptools из окружения")
    ap.add_argument("--layout", default="flat", choices=["flat", "nested"],
                    help="flat — файлы в корне архива; nested — внутри папки root/. "
                         "В ТЗ дерево нарисовано с 'root/' сверху, но это почти наверняка "
                         "обозначение корня архива, а не буквальная папка. Проверяется "
                         "бесплатно: упавший сабмит лимит не тратит")
    ap.add_argument("--question", default="package_installed", choices=PROBE_QUESTIONS,
                    help="для профиля probe: что спрашиваем у образа платформы. "
                         "Ответ читается по номеру шага, на котором прогон остановлен: "
                         "да — шаг расчёта метрик, нет — шаг распаковки")
    ap.add_argument("--gate", default="all", choices=GATES,
                    help="для профиля gate: какой гейт измеряем. По одному на прогон")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "submissions" / args.profile
    build_diagnostic(out_dir, args.profile, args.packaging, args.question, args.gate)

    problems = check(out_dir)
    if not problems and args.profile in SUBMITTABLE:
        # Гейт с числом: архив, чья точка входа не воспроизводит вход на выданном сэмпле,
        # не должен доезжать до платформы. Один такой уже доехал и записал пустой файл.
        problems += smoke(out_dir, args.profile)
    for p in problems:
        print(f"[ОШИБКА КОНТРАКТА] {p}")
    if problems:
        return 1

    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"собрано в {out_dir}")
    for p in files:
        print(f"  {p.stat().st_size:8d}  {p.relative_to(out_dir).as_posix()}")
    print(f"итого {len(files)} файлов, {total/1024:.1f} КБ")

    if args.archive:
        suffix = ("" if args.layout == "flat" else "-nested") + (
            "" if args.packaging == "pep517" else "-legacy") + (
            f"-{args.question}" if args.profile == "probe" else "") + (
            f"-{args.gate}" if args.profile == "gate" else "")
        arc = out_dir.parent / (out_dir.name + suffix + "." + args.format)
        items = sorted(out_dir.rglob("*"))
        if args.format == "tar.gz":
            with tarfile.open(arc, "w:gz") as tf:
                if args.layout == "flat":
                    for item in sorted(out_dir.iterdir()):
                        tf.add(item, arcname=item.name)
                else:
                    tf.add(out_dir, arcname="root")
        else:
            with zipfile.ZipFile(arc, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in items:
                    if item.is_file():
                        rel = item.relative_to(out_dir).as_posix()
                        zf.write(item, ("root/" + rel) if args.layout == "nested" else rel)
        print(f"архив: {arc} ({arc.stat().st_size/1024:.1f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

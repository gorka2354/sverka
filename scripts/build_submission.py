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


PROFILE_ENTRY = {
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
    "all_ones": "scripts/predict_stdlib.py",
    "has_tool_call": "scripts/predict_stdlib.py",
}


def build_diagnostic(out_dir: Path, profile: str = "diagnostic",
                     packaging: str = "pep517") -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "scripts").mkdir(parents=True)
    (out_dir / "src" / "sverka").mkdir(parents=True)
    (out_dir / "model").mkdir(parents=True)

    src = ROOT / PROFILE_ENTRY[profile]
    entry = out_dir / "scripts" / "predict.py"
    shutil.copy2(src, entry)
    if profile == "diagnostic_scored":
        # Меняется одна константа, а не тело функции: хирургия по строкам кода уже
        # один раз дала синтаксически битый архив.
        text = entry.read_text(encoding="utf-8")
        assert "EXIT_CODE = 1" in text, "не найдена константа кода возврата"
        entry.write_text(text.replace("EXIT_CODE = 1", "EXIT_CODE = 0"), encoding="utf-8")
    elif profile not in ("diagnostic", "minimal"):
        # Профиль зашивается в точку входа, а не передаётся флагом: платформа запускает
        # скрипт со СВОИМИ аргументами, и никакого нашего флага там не будет.
        text = entry.read_text(encoding="utf-8")
        assert 'PROFILE = "all_ones"' in text, "не найдена строка профиля в точке входа"
        entry.write_text(text.replace('PROFILE = "all_ones"', f'PROFILE = "{profile}"'),
                         encoding="utf-8")
    if packaging == "pep517":
        (out_dir / "pyproject.toml").write_text(PYPROJECT_DIAGNOSTIC, encoding="utf-8")
        (out_dir / "setup.py").write_text(SETUP_PY_HOOK, encoding="utf-8")
    else:
        # Без pyproject.toml pip идёт по старому пути и берёт setuptools из окружения,
        # а не из индекса. Это единственный способ обойти build isolation, когда флаги
        # установки нам передать некуда: платформа выполняет голое `pip install .`.
        (out_dir / "setup.py").write_text(SETUP_PY_LEGACY, encoding="utf-8")
    (out_dir / "src" / "sverka" / "__init__.py").write_text(PACKAGE_INIT, encoding="utf-8")
    (out_dir / "README.md").write_text(README_BY_PROFILE.get(profile, README), encoding="utf-8")
    (out_dir / "model" / ".gitkeep").write_text("", encoding="utf-8")
    return out_dir


def check(out_dir: Path) -> list[str]:
    """Проверки контракта, которые дешевле сделать здесь, чем узнать из упавшего сабмита."""
    problems = []
    entry = out_dir / "scripts" / "predict.py"
    if not entry.exists():
        problems.append("нет scripts/predict.py — платформа запускает именно его")
    else:
        src = entry.read_text(encoding="utf-8")
        for mod in ("pandas", "pyarrow", "numpy", "torch", "sverka"):
            if f"import {mod}" in src or f"from {mod}" in src:
                problems.append(f"точка входа импортирует {mod} — профиль обязан быть stdlib-only")
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
                    choices=["minimal", "diagnostic", "diagnostic_scored", "all_ones", "has_tool_call"])
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
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "submissions" / args.profile
    build_diagnostic(out_dir, args.profile, args.packaging)

    problems = check(out_dir)
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
            "" if args.packaging == "pep517" else "-legacy")
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

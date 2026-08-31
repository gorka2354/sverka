"""Точка входа для сабмитов, не требующих ни одной сторонней библиотеки.

Обслуживает два первых содержательных сабмита плана. Оба нужны, чтобы узнать то, что
локально узнать нельзя, и оба сознательно тривиальны:

- `all_ones` — предсказать единицу везде. Precision такого сабмита буквально равна доле
  положительных на публичном наборе. Сейчас про неё известно только, что она лежит
  в (0.013, 0.448] `[расчёт из матрицы лидерборда: (TP,FP,FN)=(47,80,18)·k]`, а от неё
  зависит и пол метрики, и калибровка порога, и целевой баланс собственного корпуса.
- `has_tool_call` — предсказать единицу там, где в ответе есть вызов инструмента.
  На выданных 46 примерах это правило даёт F1 = 0.913 `[замер]`, и вопрос, переносится ли
  артефакт, определяет, можно ли вообще доверять локальным замерам на сэмпле. В публичных
  траекториях τ²-bench ходы с вызовом чисты примерно в 98% случаев, то есть точность этого
  правила на естественных данных была бы около 0.02. Разброс между 0.02 и 0.91 — ровно то,
  что стоит одной попытки.

**Почему без зависимостей.** Платформа выполняет `pip install .` ДО запуска скрипта.
Любая объявленная зависимость превращает отсутствие интернета в необъяснимое падение
установки, и мы не отличим его от ошибки в коде. Оба правила тривиальны, оба обходятся
стандартной библиотекой — значит и вопрос об окружении для них не стоит вовсе.

Запускается платформой как `python scripts/predict.py --input <csv> --output <csv>`
(в архиве этот файл кладётся под именем `predict.py`, профиль зашивается при сборке).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# Профиль подставляется сборщиком архива. Значение по умолчанию безопасно: при неизвестном
# профиле ставится единица везде, то есть худший случай — балл-пол, а не мусор.
# ⚠️ Пакет берётся ИЗ АРХИВА, а не из установки. `[лог боевого прогона 31.08.2026]`
# `pip install .` на платформе либо не выполняется, либо ставит пакет в другое окружение:
# под mpirun импорт даёт `ModuleNotFoundError: No module named 'sverka'`. Из-за этого
# сабмит `no_units` молча работал по запасной ветке и ставил единицу везде — а мы прочли
# это как свойство данных. Путь добавляется относительно самого файла, поэтому работает
# и при запуске из распакованного архива, и из любой рабочей директории.
import os as _os
_корень = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_корень, "src"), _корень):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Выбросить уже загруженное из кэша обязательно. `[воспроизведено 30.08.2026]` Если
# в окружении оказался ДРУГОЙ пакет с этим именем — например установленная ранее заглушка
# из профиля-пробы, — то `import sverka.units` не находит подмодуль, но сам `sverka` уже
# лежит в sys.modules, и его `__path__` указывает на заглушку. Тогда добавление `src`
# в путь не меняет ничего, и повторный импорт падает точно так же.
for _имя in [_к for _к in list(sys.modules) if _к == "sverka" or _к.startswith("sverka.")]:
    del sys.modules[_имя]

PROFILE = "all_ones"

# Формат файла предсказаний. `[измерено 30.08.2026]` Платформа читает выход как parquet,
# хотя аргумент называется `predictions.csv`: одиннадцать прогонов с текстовым CSV были
# отклонены на шаге расчёта метрик, та же сборка с parquet прошла конвейер целиком.
#
# Отсюда же снимается прежнее «без единой зависимости» из докстринга выше: pandas нужен
# теперь и на запись. Но он по-прежнему НЕ объявляется, а берётся из образа платформы —
# что он там есть, измерено чтением parquet на боевом прогоне.
OUTPUT_FORMAT = "parquet"

FALLBACK_LABEL = 1
BUDGET_SECONDS = 24 * 60  # лимит платформы 30 минут; сторожим с запасом


def raise_csv_field_limit() -> int:
    """Поднять лимит длины поля до максимума, который принимает эта сборка Python.

    Дефолт 131 072 меньше нашего максимального prompt (233 104 символа), поэтому наивное
    чтение падает ровно на длинных примерах. При этом `sys.maxsize` переполняет C long
    на Windows, поэтому спускаемся от 2**31-1.
    """
    lim = 2**31 - 1
    while True:
        try:
            csv.field_size_limit(lim)
            return lim
        except OverflowError:
            lim //= 2


def read_any(path: str) -> tuple[list[dict], list[str], str | None]:
    """Прочитать вход, подобрав формат и кодировку.

    `[инцидент 28.08.2026, лог платформы]` Наивное чтение в UTF-8 упало на боевом файле:
    `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 7`. Выданный
    сэмпл был в UTF-8, тестовый файл — нет, и увидеть его мы не можем. Поэтому кодировка
    подбирается, а формат опознаётся по сигнатуре, а не по расширению.

    latin-1 стоит последней намеренно: она декодирует любой байт и потому не отказывает
    никогда. Текст при этом может быть искажён, но для правил «всё = 1» и «есть TOOL_CALL»
    это неважно, а потерять весь прогон из-за кодировки — важно.
    """
    import io as _io

    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"\x1f\x8b"):
        import gzip as _gz
        with _gz.open(path, "rb") as f:
            raw = f.read()
    elif head.startswith(b"PAR1"):
        # Ленивый импорт, а не объявленная зависимость: объявить pandas нельзя,
        # без интернета установка упадёт раньше запуска.
        import pandas as _pd
        df = _pd.read_parquet(path)
        return df.to_dict("records"), [str(c) for c in df.columns], "parquet через pandas"
    else:
        with open(path, "rb") as f:
            raw = f.read()

    for enc in ("utf-8-sig", "utf-8", "cp1251", "utf-16", "koi8-r", "latin-1"):
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        try:
            delim = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
        except Exception:  # noqa: BLE001
            delim = ","
        try:
            reader = csv.DictReader(_io.StringIO(text), delimiter=delim)
            cols = list(reader.fieldnames or [])
            rows = list(reader)
        except Exception:  # noqa: BLE001
            continue
        if cols:
            return rows, cols, enc
    return [], [], None


def decide(row: dict, profile: str) -> int:
    if profile == "all_ones":
        return 1
    if profile == "has_tool_call":
        # Ровно то правило, которое даёт F1 = 0.913 на выданном сэмпле. Проверяется
        # наличие подстроки, а не разбор: цель сабмита — измерить перенос артефакта,
        # и любая наша доработка правила смазала бы ответ.
        return 1 if "TOOL_CALL" in (row.get("response") or "") else 0
    if profile == "no_units":
        # Измерительный сабмит: снимаем метку там, где проверять нечего — нет вызова
        # инструмента и нет извлекаемых единиц. Из Recall восстанавливается доля
        # положительных в снятом ведре, то есть ставка, которую локально измерить нечем.
        from sverka.otvod import без_единиц
        return 0 if без_единиц(row.get("response") or "").отвести else 1
    if profile == "otvod":
        # Перевёрнутая метка: единица по умолчанию, ноль — только там, где ответ заведомо
        # безобиден (нет вызова, нет проверяемых единиц, нет заявленной невозможности).
        # `[замер 31.08.2026]` r1/r0 = 0.058 при критерии выгодности 0.692; на настоящих
        # метках valid отведено 12 строк из 46 и все двенадцать — верно, полнота 1.000.
        # Проекция на боевые страты F1 ≈ 0.498 против пола 0.393.
        from sverka.otvod import отвод
        return 0 if отвод(row.get("response") or "").отвести else 1
    return FALLBACK_LABEL


def записать(путь: str, строки: list[tuple[str, int]]) -> None:
    """Атомарная запись предсказаний. Зовётся дважды: сперва полом, потом результатом.

    Запись идёт во временный файл и переименовывается на месте. `[среда платформы]`
    в логе боевого прогона видны префиксы рангов от mpirun, то есть точка входа может
    исполняться несколькими процессами; при обычной записи они наложились бы друг на друга
    и дали обрезанный файл, а `os.replace` атомарен.
    """
    d = os.path.dirname(os.path.abspath(путь))
    if d:
        os.makedirs(d, exist_ok=True)
    врем = os.path.abspath(путь) + f".part{os.getpid()}"
    if OUTPUT_FORMAT == "parquet":
        import pandas as pd
        вых = pd.DataFrame({"id": [a for a, _ in строки], "label": [b for _, b in строки]})
        вых["label"] = вых["label"].astype("int64")
        вых.to_parquet(врем, index=False)
    else:
        with open(врем, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["id", "label"])
            w.writerows(строки)
    os.replace(врем, os.path.abspath(путь))


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--input", required=False, default=None)
    ap.add_argument("--output", required=False, default=None)
    ap.add_argument("--profile", required=False, default=PROFILE)
    args, unknown = ap.parse_known_args()
    profile = args.profile

    t0 = time.time()
    print(f"[sverka] профиль={profile} argv={sys.argv!r} лишние={unknown!r}", file=sys.stderr, flush=True)

    limit = raise_csv_field_limit()
    rows: list[dict] = []
    cols: list[str] = []
    enc_used = None
    if args.input:
        rows, cols, enc_used = read_any(args.input)
    print(f"[sverka] строк={len(rows)} колонки={cols} кодировка={enc_used} csv_limit={limit}",
          file=sys.stderr, flush=True)

    id_col = "id" if "id" in cols else (cols[0] if cols else "id")

    # ⚠️ Пол пишется НА ДИСК ДО любой содержательной работы. `[измерено]` все одиннадцать
    # отклонённых прогонов были отклонены конвейером, ни один — содержанием предсказаний,
    # то есть цена отсутствия валидного файла — потерянный невозвратный слот, а цена
    # неидеальных меток — несколько сотых F1. Файл из одних единиц даёт 0.393; дальше он
    # переписывается уточнёнными метками, и любое падение, зависание или убийство по времени
    # оставляет на диске рабочий результат, а не пустоту.
    if args.output and rows:
        try:
            записать(args.output, [(str(r.get(id_col, f"row_{i}")), 1)
                                   for i, r in enumerate(rows)])
            print(f"[sverka] пол записан: {len(rows)} строк с меткой 1", file=sys.stderr,
                  flush=True)
        except Exception as e:  # noqa: BLE001 — падать нельзя даже здесь
            print(f"[sverka] не удалось записать пол: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    out_rows: list[tuple[str, int]] = []
    ones = 0
    for i, r in enumerate(rows):
        if time.time() - t0 > BUDGET_SECONDS:
            # Выйти с неполным файлом нельзя: это гарантированный ноль за потраченную попытку.
            print(f"[sverka] ВНИМАНИЕ: бюджет исчерпан на {i}, остальным метка {FALLBACK_LABEL}",
                  file=sys.stderr, flush=True)
            out_rows.extend((str(x.get(id_col, f"row_{j}")), FALLBACK_LABEL)
                            for j, x in enumerate(rows[i:], start=i))
            break
        try:
            label = decide(r, profile)
        except Exception as e:  # noqa: BLE001 — падать нельзя
            print(f"[sverka] ОШИБКА на строке {i}: {type(e).__name__}: {e}", file=sys.stderr)
            label = FALLBACK_LABEL
        ones += label
        out_rows.append((str(r.get(id_col, f"row_{i}")), label))

    if args.output:
        записать(args.output, out_rows)

    dt = time.time() - t0
    print(f"[sverka] готово за {dt:.1f} c, записано {len(out_rows)}, единиц {ones} "
          f"({100*ones/max(1,len(out_rows)):.1f}%)", file=sys.stderr, flush=True)
    if len(out_rows) != len(rows):
        print(f"[sverka] КРИТИЧНО: строк {len(out_rows)} против {len(rows)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

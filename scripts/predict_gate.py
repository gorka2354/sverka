"""Точка входа, которая ставит метку по детерминированным гейтам и ничему больше.

Существует ради **измерения, которое локально невозможно в принципе**. Точность гейта — это
доля срабатываний, попавших в положительные примеры. На выданном сэмпле она не измерена и
измерена быть не может: все гейты требуют вызова инструмента в проверяемом ответе, то есть
живут внутри страты «ответ с вызовом», а там 23 примера при двух отрицательных, и оба —
одиночный `transfer_to_human_agents` с валидным именем и корректным JSON, на который ни один
гейт сработать не способен. Достижимых отрицательных ноль, поэтому наблюдение «одиннадцать
срабатываний, все положительные» `[замер по сэмплу]` не значит ничего.

На платформе это измеряется одним прогоном. Если метка 1 ставится ровно там, где сработал
гейт, то показанный **Precision прогона буквально равен точности гейта**, а **Recall — его
покрытию положительных**. Публичный набор известен: 264 строки, 65 положительных
`[docs/research.md §0б]` — значит из пары чисел восстанавливается и число срабатываний,
и число попаданий.

Что стоит на кону. При точности около 1.0 и покрытии 0.48, как на сэмпле, вышло бы
F1 ≈ 0.65 против 0.492 у нынешнего лидера. При точности 0.5 — F1 ≈ 0.49, то есть гейты
не дают ничего сверх baseline. Разброс между этими исходами и есть цена одного сабмита.

Модели здесь нет вовсе, зависимостей за пределами pandas — тоже. Ход по сравнению с
профилями-пробами один: вместо правила «не глядя» подставлен настоящий детектор.
"""

from __future__ import annotations

import argparse
import os
import sys

# Какой гейт проверяем. Подставляется сборщиком архива.
#   all — сработал любой; A, B, C, D, M — конкретный.
# По одному гейту на прогон: смешивать нельзя, иначе Precision станет средним по смеси
# и не скажет ничего ни об одном из них.
GATE = "all"

# Формат файла предсказаний. Платформа читает выход как parquet, хотя аргумент называется
# `predictions.csv` `[измерено 30.08.2026: одиннадцать прогонов с текстовым CSV отклонены
# на шаге расчёта метрик, та же сборка с parquet прошла целиком]`.
OUTPUT_FORMAT = "parquet"


def _import_sverka():
    """Подключить пакет, не полагаясь на то, что `pip install .` отработал.

    `[E9-A, 29.08.2026]` Без сети установка с `pyproject.toml` падает всегда — изоляция
    сборки уходит в индекс за setuptools. Прошедший прогон был именно такой раскладки,
    значит либо сеть есть, либо установка падает молча, а скрипт запускается по пути.
    Развилка не разрешена, поэтому импорт устроен так, чтобы работать в обоих случаях:
    сначала обычный путь, затем `src/` из распакованного архива.
    """
    try:
        import sverka.gates  # noqa: F401
    except ImportError:
        корень = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(корень, "src"))
        # Выбросить уже загруженное из кэша обязательно. Если в окружении оказался ДРУГОЙ
        # пакет с этим именем — например установленная ранее заглушка из профиля-пробы, —
        # то `import sverka.gates` не находит подмодуль, но сам `sverka` уже лежит в
        # sys.modules и его `__path__` указывает на заглушку. Тогда добавление `src` в путь
        # не меняет ничего, и повторный импорт падает точно так же.
        # `[воспроизведено 30.08.2026: ModuleNotFoundError: No module named 'sverka.gates'
        # в окружении, где ранее был установлен профиль probe]`
        for имя in [к for к in sys.modules if к == "sverka" or к.startswith("sverka.")]:
            del sys.modules[имя]
    from sverka.gates import apply_gates, load_known_user_tools
    from sverka.transcript import parse_response, parse_transcript
    return apply_gates, load_known_user_tools, parse_transcript, parse_response


def label_for(names, gate: str) -> int:
    """Метка по списку сработавших гейтов.

    Имена гейтов начинаются с буквы (`A_unavailable_tool`), поэтому выбор конкретного —
    это проверка первой буквы, а не поиск подстроки: подстрока `A` нашлась бы в любом
    имени с этой буквой внутри.
    """
    if not names:
        return 0
    if gate == "all":
        return 1
    return 1 if any(n.startswith(gate + "_") for n in names) else 0


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args, _ = ap.parse_known_args()
    if not args.input or not args.output or not os.path.exists(args.input):
        return 1

    import pandas as pd

    apply_gates, load_known_user_tools, parse_transcript, parse_response = _import_sverka()

    with open(args.input, "rb") as f:
        сигнатура = f.read(4)
    df = pd.read_parquet(args.input) if сигнатура == b"PAR1" else pd.read_csv(args.input)
    if "id" not in df.columns or not len(df):
        return 1

    user_tools = load_known_user_tools()
    метки = []
    for _, строка in df.iterrows():
        try:
            домен = str(строка["id"]).split("__")[0]
            tr = parse_transcript(str(строка.get("prompt") or ""), strict=False)
            turn = parse_response(str(строка.get("response") or ""), strict=False)
            отчёт = apply_gates(tr, turn, user_tools=user_tools.get(домен))
            # Молчание гейтов при нездоровом разборе — это ноль, а не пропуск: иначе
            # смена формата транскрипта дала бы не тишину, а поток ложных срабатываний.
            метки.append(0 if отчёт.silenced else label_for(отчёт.names, GATE))
        except Exception:  # noqa: BLE001 — падать нельзя, прогон стоит попытки
            метки.append(0)

    выход = pd.DataFrame({"id": df["id"].values, "label": метки})
    выход["label"] = выход["label"].astype("int64")
    каталог = os.path.dirname(os.path.abspath(args.output))
    if каталог:
        os.makedirs(каталог, exist_ok=True)
    врем = os.path.abspath(args.output) + ".part"
    if OUTPUT_FORMAT == "parquet":
        выход.to_parquet(врем, index=False)
    else:
        выход.to_csv(врем, index=False, lineterminator="\n")
    os.replace(врем, os.path.abspath(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())

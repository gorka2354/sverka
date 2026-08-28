"""Предельно простая точка входа. Существует, чтобы отладить контракт платформы вслепую.

Первый реальный сабмит упал на шаге «Статус распаковки данных / Code run», и платформа
не показывает ни текста ошибки, ни логов. Значит единственный канал информации — сам факт,
на каком шаге прогон умер. Каждый сабмит должен нести ровно один бит.

Этот файл не делает НИЧЕГО, кроме записи валидного файла предсказаний со всеми единицами:
ни диагностики, ни подпроцессов, ни сетевых проб, ни хитрых аргументов, ни поднятия лимитов.
Если он пройдёт — значит контракт освоен, и можно возвращать диагностику по одной части.
Если упадёт там же — дело не в нашем коде, а в упаковке или раскладке архива.

Единственное, что здесь всё же есть, — поднятый лимит длины поля CSV. Без него чтение
падает на длинных примерах (в выданном сэмпле максимум 233 104 символа при дефолте
131 072), и это была бы ошибка не контракта, а наша.
"""

import argparse
import csv
import os
import sys


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args, _unknown = ap.parse_known_args()

    limit = 2**31 - 1
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 2

    ids = []
    if args.input:
        with open(args.input, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            key = "id" if "id" in cols else (cols[0] if cols else None)
            for i, row in enumerate(reader):
                ids.append(str(row.get(key, i)) if key else str(i))

    if args.output:
        d = os.path.dirname(os.path.abspath(args.output))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            for x in ids:
                w.writerow([x, 1])

    print("[sverka] minimal: строк %d" % len(ids), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Устойчивое чтение входного файла платформы.

`[инцидент 28.08.2026, лог прогона]` Первый прогон, добравшийся до нашего кода, упал так:

    File "scripts/predict.py", line 41, in main
        cols = list(reader.fieldnames or [])
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 7: invalid start byte

То есть файл, который платформа передаёт в `--input`, **не в UTF-8**, хотя выданный сэмпл
был именно в нём. Предполагать кодировку входа нельзя: мы этот файл не видим и не увидим.

Модуль решает три задачи, и все три — из-за того, что вход неизвестен:

1. **Опознать формат по сигнатуре**, а не по расширению. Файл может называться `.csv`
   и быть parquet, gzip или zip: платформа обещает CSV, но обещание уже один раз
   разошлось с фактом (сэмпл выдан как parquet при обещанном CSV).
2. **Подобрать кодировку**, а не назначить. Порядок проб: utf-8 с BOM и без, cp1251,
   latin-1. Последняя декодирует что угодно и служит страховкой: лучше прочитать текст
   с испорченными буквами, чем не прочитать ничего и потерять прогон.
3. **Не падать никогда.** Возвращается результат с описанием того, что удалось, чтобы
   вызывающий код мог записать хоть какие-то предсказания и напечатать диагностику.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

# Сигнатуры форматов, которые могут прийти под видом CSV.
MAGIC = {
    b"PAR1": "parquet",
    b"\x1f\x8b": "gzip",
    b"PK\x03\x04": "zip",
    b"\xff\xfe": "utf-16le с BOM",
    b"\xfe\xff": "utf-16be с BOM",
    b"\xef\xbb\xbf": "utf-8 с BOM",
    b"\x28\xb5\x2f\xfd": "zstd",
    b"BZh": "bzip2",
    b"\x04\x22\x4d\x18": "lz4",
}

# Порядок проб кодировок. utf-8 первым, потому что так был выдан сэмпл; cp1251 — потому
# что диалоги русские и это самая частая однобайтовая кодировка для русского; latin-1
# последней, потому что она декодирует ЛЮБОЙ байт и потому никогда не отказывает.
ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "utf-16", "latin-1")


@dataclass
class ReadResult:
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    encoding: str | None = None
    delimiter: str | None = None
    magic: str | None = None
    size: int = 0
    head_hex: str = ""
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"строк={len(self.rows)} колонки={self.columns} кодировка={self.encoding} "
            f"разделитель={self.delimiter!r} формат={self.magic} размер={self.size}"
        )


def raise_csv_field_limit() -> int:
    """Поднять лимит длины поля до максимума, который принимает эта сборка Python.

    Дефолт 131 072 меньше максимального prompt выданного сэмпла (233 104), поэтому наивное
    чтение падает ровно на длинных примерах. `sys.maxsize` при этом переполняет C long
    на части платформ, поэтому спускаемся от 2**31-1.
    """
    lim = 2**31 - 1
    while True:
        try:
            csv.field_size_limit(lim)
            return lim
        except OverflowError:
            lim //= 2


def sniff_magic(head: bytes) -> str | None:
    for sig, name in MAGIC.items():
        if head.startswith(sig):
            return name
    return None


def read_table(path: str) -> ReadResult:
    """Прочитать входной файл, подобрав формат и кодировку. Не бросает исключений."""
    res = ReadResult()
    raise_csv_field_limit()

    try:
        with open(path, "rb") as f:
            head = f.read(64)
            f.seek(0, 2)
            res.size = f.tell()
    except Exception as e:  # noqa: BLE001
        res.notes.append(f"файл не открылся: {type(e).__name__}: {e}")
        return res

    res.head_hex = head[:16].hex(" ")
    res.magic = sniff_magic(head)
    if res.magic in ("parquet", "gzip", "zip", "zstd", "bzip2", "lz4"):
        res.notes.append(
            f"вход опознан как {res.magic}, а не текст — CSV-чтение пропущено"
        )
        return res

    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception as e:  # noqa: BLE001
        res.notes.append(f"файл не прочитался: {type(e).__name__}: {e}")
        return res

    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        try:
            sample = text[:65536]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                delim = dialect.delimiter
            except Exception:  # noqa: BLE001
                delim = ","
                res.notes.append("разделитель не определился, взята запятая")
            reader = csv.DictReader(io.StringIO(text), delimiter=delim)
            cols = list(reader.fieldnames or [])
            rows = list(reader)
        except Exception as e:  # noqa: BLE001
            res.notes.append(f"кодировка {enc} подошла, но разбор упал: {type(e).__name__}")
            continue
        if not cols:
            res.notes.append(f"кодировка {enc}: колонок не найдено")
            continue
        res.rows, res.columns, res.encoding, res.delimiter = rows, cols, enc, delim
        if enc == "latin-1":
            res.notes.append(
                "подошла только latin-1 — она декодирует любой байт, поэтому текст может "
                "быть искажён; для меток это обычно не важно, для разбора содержимого важно"
            )
        return res

    res.notes.append(f"ни одна кодировка не подошла, первые байты: {res.head_hex}")
    return res

"""Диагностический прогон для платформы — только стандартная библиотека.

Задача: за один заход выяснить всё, чего мы не знаем про среду исполнения, и не потратить
сабмит. Правила соревнования: упавшие решения лимит не тратят.

Четыре конструктивных требования, каждое закрывает свой способ не получить ничего
(обоснование — `inc/inc-1-diagnostics-and-gates.md`, §1):

1. **Ни одной сторонней зависимости.** Платформа выполняет `pip install .` ДО запуска
   скрипта. Если интернета нет, установка `pandas`/`pyarrow` падает раньше, чем мы что-то
   напечатаем, и «нет сети» неотличимо от «скрипт сломан». Диагностика, выясняющая наличие
   сети, не может от неё зависеть.

2. **Предсказания пишутся ПЕРВЫМИ, падение — последним.** Если платформа проигнорирует код
   возврата и засчитает прогон, слот не сгорит впустую: файл со всеми единицами — это
   запланированный сабмит «всё = 1», дающий долю положительных на публичном наборе
   (она равна Precision) и гарантированный балл-пол.

3. **Печать в stderr построчно с flush, самое ценное — последним, дубль в тексте
   исключения.** Платформы часто показывают хвост stderr или только traceback, а при
   ненулевом выходе буфер stdout может не долететь.

4. **Содержимое теста в лог не попадает.** Архив поедет на техническую экспертизу вместе
   с остальными; печатаем только производные — длины, счётчики, хеши.

Запускается платформой как `python scripts/predict.py --input <csv> --output <csv>`
(в архиве этот файл кладётся под именем `predict.py`).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import time

PROC_START = time.time()

# Код завершения. Подставляется сборщиком архива и определяет всю экономику попыток.
#
# 1 — прогон падает намеренно. По ТЗ упавшие решения лимит не тратят, значит диагностика
#     бесплатна. Но если платформа считает иначе, попытка уйдёт, а балла не будет,
#     и на долю положительных придётся тратить ещё одну.
# 0 — прогон завершается успешно. Предсказания со всеми единицами уже записаны выше,
#     поэтому он засчитывается как сабмит «всё = 1» и сразу даёт долю положительных,
#     а диагностика печатается в оба потока.
#
# Разбор по исходам: вариант с падением стоит 1 попытку в хорошем случае и 2 в плохом;
# вариант с нулём — ровно 1 всегда. Второй слабо доминирует и проигрывает лишь в одном
# сценарии: если платформа показывает логи только у упавших прогонов.
EXIT_CODE = 1

# Метки маркеров печатаются ASCII-именами: сами символы U+27E6/U+27E7 роняют вывод
# в консоли с однобайтовой кодировкой, а нам нужна не красота, а доставленный лог.
MARKERS = {
    "SYSTEM": "⟦SYSTEM⟧",
    "USER": "⟦USER⟧",
    "ASSISTANT": "⟦ASSISTANT",
    "ASSISTANT_TOOL_CALL": "⟦ASSISTANT_TOOL_CALL",
    "USER_TOOL_CALL": "⟦USER_TOOL_CALL",
    "TOOL_RESULT": "⟦TOOL_RESULT",
    "requestor_attr": "requestor=",
    "prompt_tag": "<prompt>",
    "response_tag": "<response>",
    "arrow_TOOL_CALL": "TOOL_CALL",
    "arrow_TOOL_RESPONSE": "TOOL_RESPONSE",
    "AVAILABLE_TOOLS": "[AVAILABLE TOOLS]",
    "policy_tag": "<policy>",
}

INTERESTING_PACKAGES = (
    "torch", "vllm", "transformers", "tokenizers", "accelerate", "numpy", "pandas",
    "pyarrow", "scipy", "scikit-learn", "sentencepiece", "flash-attn", "xformers",
    "nvidia", "triton", "safetensors", "huggingface",
)


def raise_csv_field_limit() -> int:
    """Поднять лимит длины поля до максимума, который принимает эта сборка Python.

    Дефолт 131 072 меньше нашего максимального prompt (233 104 символа в выданном сэмпле),
    поэтому наивное чтение падает ровно на длинных примерах. При этом `sys.maxsize`
    переполняет C long на Windows (`OverflowError`), поэтому спускаемся от 2**31-1.
    """
    lim = 2**31 - 1
    while True:
        try:
            csv.field_size_limit(lim)
            return lim
        except OverflowError:
            lim //= 2


def quantiles(values: list[int]) -> str:
    if not values:
        return "нет данных"
    s = sorted(values)

    def q(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return f"min={s[0]} p50={q(0.5)} p90={q(0.9)} max={s[-1]} sum={sum(s)}"


def run(cmd: list[str], timeout: int = 20) -> str:
    """Выполнить внешнюю команду, вернув однострочный результат или причину отказа."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        return out.strip() or f"(пусто, код {p.returncode})"
    except FileNotFoundError:
        return "(команда не найдена)"
    except subprocess.TimeoutExpired:
        return f"(таймаут {timeout} c)"
    except Exception as e:  # noqa: BLE001 — диагностика не имеет права падать
        return f"({type(e).__name__}: {e})"


def probe_network() -> str:
    """Есть ли исходящая сеть. Три цели, короткий таймаут, без DNS там, где можно."""
    results = []
    for host, port in (("pypi.org", 443), ("1.1.1.1", 53), ("huggingface.co", 443)):
        t0 = time.time()
        try:
            with socket.create_connection((host, port), timeout=3):
                results.append(f"{host}:{port}=OK({(time.time()-t0)*1000:.0f}ms)")
        except Exception as e:  # noqa: BLE001
            results.append(f"{host}:{port}={type(e).__name__}")
    return " ".join(results)


def container_uptime() -> str:
    """Время с запуска контейнера — единственный способ узнать, входит ли `pip install`
    в лимит 30 минут. Внутрипроцессный таймер стартует уже ПОСЛЕ установки и на этот
    вопрос не отвечает.
    """
    parts = []
    try:
        with open("/proc/uptime", encoding="utf-8") as f:
            parts.append(f"proc_uptime={float(f.read().split()[0]):.1f}s")
    except Exception:  # noqa: BLE001 — на не-Linux файла просто нет
        parts.append("proc_uptime=n/a")
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parts.append(f"archive_age={time.time() - os.path.getmtime(root):.1f}s")
    except Exception as e:  # noqa: BLE001
        parts.append(f"archive_age=({type(e).__name__})")
    parts.append(f"in_process={time.time() - PROC_START:.1f}s")
    return " ".join(parts)


def try_import(name: str) -> str:
    try:
        mod = __import__(name)
        return f"{name}={getattr(mod, '__version__', 'без __version__')}"
    except Exception as e:  # noqa: BLE001
        return f"{name}=НЕТ ({type(e).__name__})"


def read_rows(path: str) -> tuple[list[dict], list[str], str]:
    """Прочитать вход стандартной библиотекой. Возвращает строки, колонки и заметки."""
    notes = []
    limit = raise_csv_field_limit()
    notes.append(f"csv.field_size_limit={limit}")
    with open(path, encoding="utf-8", newline="") as f:
        sample = f.read(65536)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            notes.append(f"delimiter={dialect.delimiter!r}")
        except Exception:  # noqa: BLE001
            dialect = csv.excel  # type: ignore[assignment]
            notes.append("delimiter=не определён, взята запятая")
        reader = csv.DictReader(f, dialect=dialect)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, cols, "; ".join(notes)


def write_all_ones(rows: list[dict], cols: list[str], out_path: str) -> str:
    """Записать валидные предсказания ДО любой диагностики.

    Это страховка на случай, что платформа проигнорирует ненулевой код возврата: тогда
    прогон засчитается как сабмит «всё = 1», а не как потраченный впустую слот.
    """
    id_col = "id" if "id" in cols else (cols[0] if cols else "id")
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, r in enumerate(rows):
            w.writerow([r.get(id_col, f"row_{i}"), 1])
    return f"записано {len(rows)} предсказаний (все = 1), колонка id взята из {id_col!r}"


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--input", required=False, default=None)
    ap.add_argument("--output", required=False, default=None)
    args, unknown = ap.parse_known_args()

    lines: list[str] = []

    def emit(s: str, stream) -> None:
        try:
            print(s, file=stream, flush=True)
        except Exception:  # noqa: BLE001 — кодировка консоли не должна ронять прогон
            try:
                print(s.encode("ascii", "replace").decode(), file=stream, flush=True)
            except Exception:  # noqa: BLE001
                pass

    def say(s: str) -> None:
        """Построчно в stderr с flush — его платформы показывают чаще и он не буферизуется.

        Дублируется всё это не построчно, а целиком: одним блоком в stdout в самом конце
        и текстом финального исключения. Так лог не удваивается, но остаётся три
        независимых шанса, что он дойдёт.
        """
        lines.append(s)
        emit(s, sys.stderr)

    say("=" * 72)
    say("[sverka] ДИАГНОСТИКА СРЕДЫ. Прогон намеренно завершается с ненулевым кодом.")
    say(f"[sverka] argv={sys.argv!r} неизвестные аргументы={unknown!r}")

    # --- 1. Вход и предсказания. Сначала файл, потом всё остальное. ---
    rows: list[dict] = []
    cols: list[str] = []
    if args.input:
        try:
            rows, cols, notes = read_rows(args.input)
            say(f"[вход] строк={len(rows)} колонки={cols}")
            say(f"[вход] {notes}")
            try:
                h = hashlib.sha256()
                with open(args.input, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                say(f"[вход] размер={os.path.getsize(args.input)} sha256={h.hexdigest()[:16]}")
            except Exception as e:  # noqa: BLE001
                say(f"[вход] хеш не посчитан: {type(e).__name__}")
        except Exception as e:  # noqa: BLE001
            say(f"[вход] ЧТЕНИЕ УПАЛО: {type(e).__name__}: {e}")
    else:
        say("[вход] --input не передан")

    if args.output:
        try:
            say(f"[выход] {write_all_ones(rows, cols, args.output)}")
        except Exception as e:  # noqa: BLE001
            say(f"[выход] ЗАПИСЬ УПАЛА: {type(e).__name__}: {e}")
    else:
        say("[выход] --output не передан")

    # --- 2. Форма данных: только производные, без содержимого. ---
    if rows:
        pcol = "prompt" if "prompt" in cols else None
        rcol = "response" if "response" in cols else None
        if pcol:
            say(f"[длины] prompt {quantiles([len(r.get(pcol) or '') for r in rows])}")
        if rcol:
            say(f"[длины] response {quantiles([len(r.get(rcol) or '') for r in rows])}")

        hist = {k: 0 for k in MARKERS}
        docs = 0
        for r in rows:
            blob = (r.get(pcol) or "") + "\n" + (r.get(rcol) or "") if pcol or rcol else " ".join(
                str(v) for v in r.values()
            )
            docs += 1
            for name, token in MARKERS.items():
                if token in blob:
                    hist[name] += 1
        say(f"[маркеры] документов={docs} (значение = в скольких строках встретился маркер)")
        say("[маркеры] " + " ".join(f"{k}={v}" for k, v in hist.items()))

        if "id" in cols:
            dom: dict[str, int] = {}
            for r in rows:
                d = str(r.get("id") or "").split("__")[0][:32]
                dom[d] = dom.get(d, 0) + 1
            say("[домены] " + " ".join(f"{k}={v}" for k, v in sorted(dom.items())))
            say(f"[домены] уникальных диалогов={len({str(r.get('id') or '').split('::')[0] for r in rows})}")

    # --- 3. Среда. ---
    say(f"[python] {sys.version.replace(chr(10), ' ')}")
    say(f"[python] executable={sys.executable} platform={sys.platform}")
    say(f"[время] {container_uptime()}")
    try:
        du = shutil.disk_usage(os.getcwd())
        say(f"[диск] всего={du.total/1e9:.1f}ГБ свободно={du.free/1e9:.1f}ГБ cwd={os.getcwd()}")
    except Exception as e:  # noqa: BLE001
        say(f"[диск] ({type(e).__name__})")
    say(f"[сеть] {probe_network()}")

    pip_out = run([sys.executable, "-m", "pip", "list", "--disable-pip-version-check"], timeout=60)
    pkgs = [ln for ln in pip_out.splitlines() if any(p in ln.lower() for p in INTERESTING_PACKAGES)]
    say(f"[pip] всего строк={len(pip_out.splitlines())}, интересные:")
    for ln in pkgs[:40]:
        say(f"[pip]   {ln.strip()}")
    if not pkgs:
        say("[pip]   ни одного интересного пакета не найдено")

    smi = run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        timeout=20,
    )
    say(f"[gpu] nvidia-smi: {smi[:300]}")

    # Самое ценное — последним: если платформа покажет только хвост, он должен быть здесь.
    say("[ИТОГ] " + " | ".join(try_import(m) for m in ("torch", "vllm", "transformers", "numpy")))
    say("=" * 72)

    blob = "\n".join(lines)
    # Второй канал: весь блок одним куском в stdout.
    emit("\n[sverka] ПОЛНЫЙ БЛОК ДИАГНОСТИКИ (дубль):\n" + blob, sys.stdout)
    if EXIT_CODE == 0:
        emit("[sverka] выходим с кодом 0: предсказания записаны, прогон должен засчитаться",
             sys.stderr)
        raise SystemExit(0)
    # Третий канал: текст исключения. Если из всего лога дойдёт только traceback — дойдёт и он.
    raise SystemExit("ДИАГНОСТИКА (намеренное падение, сабмит не должен засчитаться)\n" + blob)


if __name__ == "__main__":
    main()

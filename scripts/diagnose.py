"""Диагностический прогон для платформы — только стандартная библиотека, ни одного импорта
из нашего пакета.

## Что уже выяснено прогонами на платформе

`[лог прогона 28.08.2026]` Контракт архива освоен: платформа находит `scripts/predict.py`,
ставит пакет и запускает его. Шаг «08. Статус распаковки данных / Code run» — это и есть
выполнение нашего кода, а вовсе не распаковка архива. Среда: Cloud.ru, `/home/jovyan`,
conda, Python 3.12, запуск под `mpirun` (то есть решение стартует как MPI-задача).

И главное: **входной файл не в UTF-8.** Наивное чтение упало так —

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 7: invalid start byte

Выданный сэмпл был в UTF-8, тестовый файл — нет. Предполагать кодировку входа нельзя,
поэтому здесь она подбирается, а формат опознаётся по сигнатуре, а не по расширению.

## Правила, из-за которых код выглядит именно так

**Ни одного стороннего импорта.** Платформа выполняет `pip install .` до запуска скрипта.
Любая объявленная зависимость превращает отсутствие интернета в необъяснимое падение
установки. И ни одного импорта из `sverka`: если пакет по какой-то причине не встанет,
диагностика обязана всё равно отработать.

**Ни одной строки содержимого теста в лог.** Печатаются производные: сигнатура первых
байт, длины, счётчики, имена колонок. Архив поедет на техническую экспертизу.

**Предсказания пишутся до печати.** Если файл удалось прочитать, выходим нулём и прогон
засчитывается; если нет — единицей, и попытка возвращается.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import socket
import subprocess
import sys
import time

PROC_START = time.time()

MAGIC = {
    b"PAR1": "parquet",
    b"\x1f\x8b": "gzip",
    b"PK\x03\x04": "zip",
    b"\xff\xfe": "utf-16le с BOM",
    b"\xfe\xff": "utf-16be с BOM",
    b"\xef\xbb\xbf": "utf-8 с BOM",
    b"\x28\xb5\x2f\xfd": "zstd",
    b"BZh": "bzip2",
}

# utf-8 первым — так был выдан сэмпл; cp1251 — диалоги русские; latin-1 последней,
# потому что она декодирует любой байт и потому никогда не отказывает.
ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "utf-16", "koi8-r", "latin-1")

INTERESTING = ("torch", "vllm", "transformers", "tokenizers", "accelerate", "numpy",
               "pandas", "pyarrow", "sentencepiece", "nvidia", "triton", "safetensors")


def raise_csv_field_limit() -> int:
    lim = 2**31 - 1
    while True:
        try:
            csv.field_size_limit(lim)
            return lim
        except OverflowError:
            lim //= 2


def run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return ((p.stdout or "") + (p.stderr or "")).strip() or f"(пусто, код {p.returncode})"
    except FileNotFoundError:
        return "(команда не найдена)"
    except Exception as e:  # noqa: BLE001
        return f"({type(e).__name__})"


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args, unknown = ap.parse_known_args()

    lines: list[str] = []

    def emit(s, stream):
        try:
            print(s, file=stream, flush=True)
        except Exception:  # noqa: BLE001
            try:
                print(s.encode("ascii", "replace").decode(), file=stream, flush=True)
            except Exception:  # noqa: BLE001
                pass

    def say(s):
        lines.append(s)
        emit(s, sys.stderr)

    say("=" * 70)
    say(f"[sverka] argv={sys.argv!r} неизвестные аргументы={unknown!r}")
    say(f"[sverka] cwd={os.getcwd()} лимит поля csv={raise_csv_field_limit()}")

    rows: list[dict] = []
    cols: list[str] = []
    enc_used = None

    # --- ОПОЗНАНИЕ ФОРМАТА -----------------------------------------------------------
    if args.input:
        try:
            with open(args.input, "rb") as f:
                head = f.read(64)
            size = os.path.getsize(args.input)
            magic = next((n for sig, n in MAGIC.items() if head.startswith(sig)), None)
            say(f"[вход] путь={args.input} размер={size}")
            say(f"[вход] первые 16 байт: {head[:16].hex(' ')}")
            say(f"[вход] сигнатура: {magic or 'не опознана, вероятно текст'}")
            say(f"[вход] неascii-байтов в первых 64: {sum(1 for b in head if b > 127)}")

            raw = None
            if magic == "gzip":
                # gzip лежит в стандартной библиотеке — распаковываем и читаем как текст.
                import gzip as _gz
                try:
                    with _gz.open(args.input, "rb") as f:
                        raw = f.read()
                    say("[вход] gzip распакован средствами стандартной библиотеки")
                except Exception as e:  # noqa: BLE001
                    say(f"[вход] gzip не распаковался: {type(e).__name__}")
            elif magic in ("parquet", "zip", "zstd", "bzip2"):
                # Ленивый импорт, а НЕ объявленная зависимость: если pandas есть
                # в окружении платформы — воспользуемся, если нет — честно скажем.
                # Объявлять его в pyproject нельзя: без интернета установка упадёт.
                say(f"[вход] ЭТО НЕ ТЕКСТ: {magic}. Пробуем ленивый импорт pandas.")
                try:
                    import pandas as _pd
                    _df = _pd.read_parquet(args.input) if magic == "parquet" else _pd.read_csv(args.input)
                    cols = [str(c) for c in _df.columns]
                    rows = _df.to_dict("records")
                    enc_used = f"{magic} через pandas {_pd.__version__}"
                    say(f"[вход] прочитано pandas: строк={len(rows)} колонки={cols}")
                except Exception as e:  # noqa: BLE001
                    say(f"[вход] pandas недоступен или не смог: {type(e).__name__}: {e}")

            if raw is None and not rows and magic not in ("parquet", "zip", "zstd", "bzip2"):
                with open(args.input, "rb") as f:
                    raw = f.read()

            if raw is not None:
                for e in ENCODINGS:
                    try:
                        text = raw.decode(e)
                    except (UnicodeDecodeError, LookupError) as err:
                        say(f"[кодировка] {e:10s} НЕТ ({type(err).__name__})")
                        continue
                    try:
                        try:
                            delim = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
                        except Exception:  # noqa: BLE001
                            delim = ","
                        r = csv.DictReader(io.StringIO(text), delimiter=delim)
                        c = list(r.fieldnames or [])
                        rr = list(r)
                    except Exception as err:  # noqa: BLE001
                        say(f"[кодировка] {e:10s} декодировала, но разбор упал: {type(err).__name__}")
                        continue
                    say(f"[кодировка] {e:10s} ДА, разделитель={delim!r}, колонок={len(c)}, строк={len(rr)}")
                    if c and not cols:
                        rows, cols, enc_used = rr, c, e
                say(f"[вход] выбрана кодировка: {enc_used}")
                say(f"[вход] колонки: {cols}")
                say(f"[вход] строк: {len(rows)}")
        except Exception as e:  # noqa: BLE001
            say(f"[вход] ЧТЕНИЕ УПАЛО: {type(e).__name__}: {e}")
    else:
        say("[вход] --input не передан")

    # --- ПРОИЗВОДНЫЕ ПО ДАННЫМ, без единой строки содержимого ------------------------
    if rows:
        pcol = "prompt" if "prompt" in cols else None
        rcol = "response" if "response" in cols else None
        for name, col in (("prompt", pcol), ("response", rcol)):
            if col:
                L = sorted(len(str(r.get(col) or "")) for r in rows)
                say(f"[длины] {name}: min={L[0]} p50={L[len(L)//2]} "
                    f"p90={L[int(0.9*len(L))]} max={L[-1]} сумма={sum(L)}")
        blob_cols = [c for c in (pcol, rcol) if c]
        marks = ["⟦SYSTEM⟧", "⟦USER⟧", "⟦ASSISTANT", "⟦ASSISTANT_TOOL_CALL", "⟦TOOL_RESULT",
                 "requestor=", "<prompt>", "TOOL_CALL", "TOOL_RESPONSE", "[AVAILABLE TOOLS]"]
        hist = {m: 0 for m in marks}
        for r in rows:
            blob = " ".join(str(r.get(c) or "") for c in blob_cols)
            for m in marks:
                if m in blob:
                    hist[m] += 1
        say("[маркеры] " + " ".join(f"{m.strip('⟦⟧[]')}={v}" for m, v in hist.items()))
        if "id" in cols:
            dom: dict[str, int] = {}
            for r in rows:
                d = str(r.get("id") or "").split("__")[0][:24]
                dom[d] = dom.get(d, 0) + 1
            say("[домены] " + " ".join(f"{k}={v}" for k, v in sorted(dom.items())))

    # --- ПРЕДСКАЗАНИЯ: пишем ДО остальной диагностики ---------------------------------
    wrote = 0
    if args.output:
        try:
            d = os.path.dirname(os.path.abspath(args.output))
            if d:
                os.makedirs(d, exist_ok=True)
            key = "id" if "id" in cols else (cols[0] if cols else None)
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "label"])
                for i, r in enumerate(rows):
                    w.writerow([str(r.get(key, i)) if key else str(i), 1])
                    wrote += 1
            say(f"[выход] записано {wrote} предсказаний (все = 1) в {args.output}")
        except Exception as e:  # noqa: BLE001
            say(f"[выход] ЗАПИСЬ УПАЛА: {type(e).__name__}: {e}")

    # --- СРЕДА ------------------------------------------------------------------------
    say(f"[python] {sys.version.replace(chr(10), ' ')}")
    say(f"[python] executable={sys.executable} platform={sys.platform}")
    try:
        with open("/proc/uptime", encoding="utf-8") as f:
            say(f"[время] контейнер живёт {float(f.read().split()[0]):.0f} c, "
                f"в процессе {time.time()-PROC_START:.1f} c")
    except Exception:  # noqa: BLE001
        say(f"[время] в процессе {time.time()-PROC_START:.1f} c")
    try:
        du = shutil.disk_usage(os.getcwd())
        say(f"[диск] всего={du.total/1e9:.0f}ГБ свободно={du.free/1e9:.0f}ГБ")
    except Exception:  # noqa: BLE001
        pass
    net = []
    for host, port in (("pypi.org", 443), ("1.1.1.1", 53)):
        t0 = time.time()
        try:
            with socket.create_connection((host, port), timeout=3):
                net.append(f"{host}=OK({(time.time()-t0)*1000:.0f}ms)")
        except Exception as e:  # noqa: BLE001
            net.append(f"{host}={type(e).__name__}")
    say("[сеть] " + " ".join(net))
    say(f"[gpu] {run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])[:200]}")
    pip_out = run([sys.executable, "-m", "pip", "list", "--disable-pip-version-check"], timeout=60)
    pk = [ln.strip() for ln in pip_out.splitlines() if any(p in ln.lower() for p in INTERESTING)]
    say(f"[pip] всего пакетов~{len(pip_out.splitlines())}, интересные: {pk[:25]}")
    say("[ИТОГ] " + " | ".join(
        f"{m}={getattr(__import__(m), '__version__', '?')}" if _try(m) else f"{m}=НЕТ"
        for m in ("torch", "vllm", "transformers", "numpy")))
    say("=" * 70)

    emit("\n[sverka] ПОЛНЫЙ БЛОК (дубль в stdout):\n" + "\n".join(lines), sys.stdout)

    if wrote:
        say("[sverka] предсказания записаны — выходим нулём, прогон должен засчитаться")
        return 0
    raise SystemExit("ДИАГНОСТИКА: прочитать вход не удалось\n" + "\n".join(lines))


def _try(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())

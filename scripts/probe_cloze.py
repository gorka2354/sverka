"""Слепое заполнение пропуска на ЧИСТЫХ единицах: умеет ли модель вообще извлекать.

Разделяет развилку, к которой свёлся весь день:

- **A.** модель не ДОСТАЁТ значение из тринадцати тысяч токенов — виноваты длина окна и
  четыре бита квантования. Лечится коротким срезом и другой моделью;
- **B.** модель не РЕШАЕТ задачу ни в каком виде — тогда никакая архитектура промпта
  не спасёт, и надо строить политику сабмитов вокруг пола 0.393.

Замер устроен так, чтобы верный ответ был **заведомо известен**: берутся единицы из ЧИСТЫХ
ходов, то есть из ответов, которые агент написал верно. Значит то, что он написал, и есть
правильный ответ, и никакой разметки не требуется. Спрашивать при этом можно только про те
единицы, чьё значение реально попало в срез — иначе мы мерили бы чанкер.

⚠️ Гейты условные, знаменатель берётся из `probe_slice_ceiling.py`. `[замер]` потолок
доставки на русском — 0.962, поэтому гейт 0.85 проверяет модель. Безусловный гейт 0.80 при
скупом чанкере (донесено 0.755) проверял бы отбор кусков и ничего не сказал бы о модели.

    python scripts/probe_cloze.py --limit 150
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.ask import SCHEMA, проверить, собрать  # noqa: E402
from sverka.slices import отобрать  # noqa: E402
from sverka.units import извлечь, есть_в_тексте  # noqa: E402

# ── ГЕЙТЫ. Записаны до прогона. Знаменатель — единицы, чьё значение ЕСТЬ в срезе. ────────
ГЕЙТЫ = {
    "подтверждено_из_доступных": (0.85, "больше",
                                  "потолок доставки 0.962; ниже 0.85 — модель не извлекает"),
    "противоречит_из_доступных": (0.05, "меньше",
                                  "прямая доля ложных сильнейшего канала"),
    "доля_непроверяемых": (0.10, "меньше",
                           "принудительная схема обязана удерживать модель на извлечении"),
}


def спросить(host, model, system, user, timeout, думать):
    тело = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False, "format": SCHEMA,
            "options": {"temperature": 0.0, "num_predict": 200}}
    # `think` кладётся в запрос ТОЛЬКО когда его просят выключить. `[замер 30.08.2026]`
    # На qwen3.5 без него ответ пустой, на gpt-oss:20b — пустой ИМЕННО с ним. Универсального
    # значения нет, поэтому это аргумент, а не константа.
    if not думать:
        тело["think"] = False
    req = urllib.request.Request(f"{host}/api/chat",
                                 data=json.dumps(тело).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        о = json.loads(r.read().decode("utf-8"))
    текст = (о.get("message", {}).get("content", "") or "")
    текст = текст.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    return текст, time.time() - t0


def собрать_задания(valid, corpus, предел_корпуса):
    """Единицы из чистых ходов текстовой страты, с разрезом по языку."""
    задания = []
    if Path(valid).exists():
        import pandas as pd
        d = pd.read_parquet(valid)
        # Только ЧИСТЫЕ ходы: у испорченного правильный ответ неизвестен без разметки.
        чистые = d[(d["label"] == 0)
                   & (~d["response"].astype(str).str.contains("TOOL_CALL", regex=False))]
        for _, r in чистые.iterrows():
            задания += [("RU", str(r["prompt"]), е) for е in извлечь(str(r["response"]))]
    if Path(corpus).exists():
        строки = [json.loads(s) for s in
                  Path(corpus).read_text(encoding="utf-8").splitlines() if s.strip()]
        for r in строки[:предел_корпуса]:
            задания += [("EN", str(r["prompt"]), е) for е in извлечь(str(r["response"]))]
    return задания


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--think", action="store_true",
                    help="НЕ выключать рассуждение (нужно для gpt-oss:20b)")
    ap.add_argument("--valid", default="data/raw/valid.parquet")
    ap.add_argument("--corpus", default="data/text_negatives.jsonl")
    ap.add_argument("--corpus-limit", type=int, default=25)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out", default="data/cloze_clean.jsonl")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    задания = собрать_задания(args.valid, args.corpus, args.corpus_limit)
    # Спрашивать только там, где верный ответ доступен: иначе меряется чанкер.
    годные = []
    for язык, контекст, ед in задания:
        срез = отобрать(контекст, ед)
        if срез.пуст or not есть_в_тексте(ед, срез.текст()):
            continue
        годные.append((язык, ед, срез))
    print(f"единиц всего {len(задания)}, годных для честного вопроса {len(годные)}",
          file=sys.stderr, flush=True)

    # Чередуем языки, чтобы обрыв прогона не оставил один язык без данных.
    ru = [x for x in годные if x[0] == "RU"]
    en = [x for x in годные if x[0] == "EN"]
    перемешано = [x for пара in zip(ru + [None] * len(en), en + [None] * len(ru))
                  for x in пара if x is not None][: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, (язык, ед, срез) in enumerate(перемешано, 1):
            в = собрать(ед, срез.текст())
            try:
                текст, сек = спросить(args.host, args.model, в.system, в.user,
                                      args.timeout, args.think)
                ответ = json.loads(текст) if текст.strip().startswith("{") else {}
                исход = проверить(ответ, в)
                ошибка = ""
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                ответ, исход, сек, ошибка = {}, "непроверяемо", 0.0, type(e).__name__
            f.write(json.dumps({
                "язык": язык, "вид": ед.вид, "истина": ед.значение,
                "исход": исход, "ответ": ответ.get("значение", ""),
                "цитата": str(ответ.get("цитата", ""))[:200],
                "срез_симв": срез.символов, "секунд": round(сек, 1), "ошибка": ошибка,
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(перемешано)}] {язык} {ед.вид:<13} истина={ед.значение[:18]:<18} "
                  f"ответ={str(ответ.get('значение',''))[:18]:<18} {исход:<14} {сек:.1f}с {ошибка}",
                  file=sys.stderr, flush=True)
    отчёт(out)
    return 0


def отчёт(path: Path) -> None:
    строки = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not строки:
        print("пусто"); return
    print(f"\n{'='*84}\nСЛЕПОЕ ЗАПОЛНЕНИЕ ПРОПУСКА: {len(строки)} вопросов\n{'='*84}")
    for язык in ("RU", "EN", "все"):
        гр = строки if язык == "все" else [x for x in строки if x["язык"] == язык]
        if not гр:
            continue
        c = collections.Counter(x["исход"] for x in гр)
        n = len(гр)
        факт = {
            "подтверждено_из_доступных": c["подтверждено"] / n,
            "противоречит_из_доступных": c["противоречит"] / n,
            "доля_непроверяемых": c["непроверяемо"] / n,
        }
        print(f"\n  ── {язык} ── n={n}, среднее {sum(x['секунд'] for x in гр)/n:.1f} с")
        print(f"     исходы: {dict(c)}")
        for имя, (порог, сторона, зачем) in ГЕЙТЫ.items():
            v = факт[имя]
            ок = (v < порог) if сторона == "меньше" else (v > порог)
            метка = "ок" if ок else "ПРОВАЛ"
            звёздочка = "" if язык != "все" else "   (гейт считается по языку, не по смеси)"
            print(f"     {имя:<28} {v:>6.3f}  нужно {сторона} {порог}  {метка}{звёздочка}")
    ru = [x for x in строки if x["язык"] == "RU"]
    if ru:
        c = collections.Counter(x["исход"] for x in ru)
        решение = ("A — модель извлекает, дело было в длине окна и битах"
                   if c["подтверждено"] / len(ru) > 0.85
                   else "B — модель не извлекает даже из короткого среза с готовым ответом")
        print(f"\n  РАЗВИЛКА по русскому: {решение}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        отчёт(Path(sys.argv[2])); raise SystemExit(0)
    raise SystemExit(main())

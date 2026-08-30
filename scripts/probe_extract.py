"""Извлекающая сверка целиком: единицы ответа → срез → узкий вопрос → вердикт нашим кодом.

Зачем. `[замер 31.08.2026]` Судья-оценщик опровергнут: на коротких ответах, из которых
боевой набор состоит на 91%, он поймал 0 из 11 положительных во всех четырёх категориях
сразу. Диагноз — он проверяет ответ НА ВНУТРЕННЮЮ непротиворечивость: агент пишет
«$226.49, возврат $30.18» при каталожных $242.92, а судья ставит A, потому что
256.67 − 226.49 = 30.18 сходится внутри самого ответа, и уликой приводит цитату из него же.

Здесь постановка другая и вопрос модели узкий: «какое значение стоит на месте пропуска
по данным среза». Вердикт выводится НЕ из слов модели, а проверкой нашим кодом — цитата
обязана дословно найтись в срезе, значение обязано найтись в цитате (`ask.py::проверить`).

Что этот прогон обязан выяснить, и что его опровергнет:

1. **Разделяет ли счёт «противоречит» положительные и отрицательные.** `[замер probe_cloze]`
   доля ложных на ЕДИНИЦУ равна 0.130, а медиана единиц на ответ — семь. Значит правило
   «хоть одно противоречие → 1» даёт ложную тревогу с вероятностью 0.62 и негодно заведомо.
   Порог «≥3» опускает её до 0.052, но требует трёх дефектов там, где порча обычно одна.
   **Опровержение всей линии:** если AUC счёта противоречий на настоящих метках ≤ 0.6,
   извлекающая сверка не работает так же, как не работал оценщик, и остаётся принять
   пол 0.393.
2. **Держится ли разделение в СЛОЕ КОРОТКИХ ответов.** Судья умирал именно там. Считать
   по слоям обязательно: корпус смещён (медиана ответа 942 символа против 348 у боевых).

⚠️ Чего этот прогон НЕ доказывает. Фильтр «значение агента само лежит в срезе» на наборе
`probe_cloze` снимал 12 ложных тревог из 12 — и это тавтология: задания там отбирались
условием «значение агента есть в срезе», поэтому фильтр снимал всё подряд, включая
настоящие срабатывания. Проверять такой фильтр можно только здесь, на настоящих метках.

    python scripts/probe_extract.py --data data/extract_set.jsonl --out data/extract.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka import ollama  # noqa: E402
from sverka.ask import SCHEMA, проверить, собрать  # noqa: E402
from sverka.slices import отобрать  # noqa: E402
from sverka.transcript import parse_response  # noqa: E402
from sverka.units import есть_в_тексте, извлечь  # noqa: E402


def проверить_ответ(host, model, num_ctx, timeout, prompt, response, предел_единиц):
    """Все единицы одного ответа. Возвращает список исходов по единицам."""
    turn = parse_response(response, strict=False)
    единицы = извлечь(turn.text or response)[:предел_единиц]
    исходы = []
    for ед in единицы:
        срез = отобрать(prompt, ед)
        if срез.пуст:
            исходы.append({"вид": ед.вид, "значение": ед.значение, "исход": "нет_среза",
                           "в_срезе": False, "секунд": 0.0})
            continue
        # Признак пишется ДО вопроса и независимо от него: лежит ли то, что написал агент,
        # в отобранном куске контекста. Он же — кандидат в фильтры ложных тревог, но
        # проверять его можно только на настоящих метках (см. предупреждение в шапке).
        в_срезе = есть_в_тексте(ед, срез.текст())
        в = собрать(ед, срез.текст())
        try:
            о = ollama.спросить(host, model, в.system, в.user, num_ctx=num_ctx,
                                формат=SCHEMA, num_predict=200, думать=False,
                                timeout=timeout)
            ответ = json.loads(о.текст) if о.текст.strip().startswith("{") else {}
            исход = проверить(ответ, в)
            исходы.append({"вид": ед.вид, "значение": ед.значение, "исход": исход,
                           "в_срезе": в_срезе, "нашлось": str(ответ.get("значение", ""))[:60],
                           "срез_симв": срез.символов, "дошло": о.дошло,
                           "секунд": round(о.секунд, 1)})
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError,
                ollama.ВходОбрезан) as e:
            исходы.append({"вид": ед.вид, "значение": ед.значение, "исход": "ошибка",
                           "в_срезе": в_срезе, "ошибка": f"{type(e).__name__}: {e}"[:120],
                           "секунд": 0.0})
    return исходы


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--data", required=True, help="jsonl с id/prompt/response/label")
    ap.add_argument("--out", default="data/extract.jsonl")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--единиц", type=int, default=8,
                    help="предел единиц на ответ: 540 строк × N × 2.5 с должны влезть в бюджет")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    строки = [json.loads(s) for s in
              Path(args.data).read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.limit:
        строки = строки[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    сделано = set()
    if out.exists():
        чужие = 0
        for s in out.read_text(encoding="utf-8").splitlines():
            try:
                z = json.loads(s)
            except Exception:  # noqa: BLE001
                continue
            if z.get("модель") == args.model and z.get("окно") == args.num_ctx:
                сделано.add(z["id"])
            else:
                чужие += 1
        if чужие:
            print(f"[ОТКАЗ] в {out} {чужие} строк с другими условиями — возьмите новый --out",
                  file=sys.stderr)
            return 2
        print(f"[возобновление] готово: {len(сделано)}", file=sys.stderr, flush=True)

    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(строки, 1):
            if r["id"] in сделано:
                continue
            исходы = проверить_ответ(args.host, args.model, args.num_ctx, args.timeout,
                                     str(r["prompt"]), str(r["response"]), args.единиц)
            c = collections.Counter(x["исход"] for x in исходы)
            f.write(json.dumps({
                "id": r["id"], "label": int(r["label"]), "домен": r.get("domain", ""),
                "единиц": len(исходы), "противоречит": c["противоречит"],
                "подтверждено": c["подтверждено"], "непроверяемо": c["непроверяемо"],
                "нет_опоры": c["нет_опоры"], "исходы": исходы,
                "ответ_симв": len(str(r["response"])),
                "модель": args.model, "окно": args.num_ctx,
                "секунд": round(sum(x["секунд"] for x in исходы), 1),
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(строки)}] {r['id'][:42]:<42} метка={r['label']} "
                  f"единиц={len(исходы)} против={c['противоречит']} "
                  f"подтв={c['подтверждено']} {sum(x['секунд'] for x in исходы):.0f}с",
                  file=sys.stderr, flush=True)
    отчёт(out)
    return 0


def отчёт(path: Path) -> None:
    строки = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not строки:
        print("пусто")
        return
    поз = [s for s in строки if s["label"] == 1]
    отр = [s for s in строки if s["label"] == 0]
    print(f"\n{'='*78}\nИЗВЛЕКАЮЩАЯ СВЕРКА: {len(строки)} ответов "
          f"({len(поз)} положительных, {len(отр)} отрицательных)\n{'='*78}")
    вс = sum(s["единиц"] for s in строки)
    print(f"  единиц всего {вс}, среднее на ответ {вс/len(строки):.1f}, "
          f"время {sum(s['секунд'] for s in строки)/60:.1f} мин "
          f"({sum(s['секунд'] for s in строки)/max(вс,1):.1f} с на единицу)")
    if not поз or not отр:
        print("  одной из групп нет — разделение не считается")
        return

    def auc(ключ):
        a = [ключ(s) for s in поз]
        b = [ключ(s) for s in отр]
        w = sum((1 if x > y else 0.5 if x == y else 0) for x in a for y in b)
        return w / (len(a) * len(b))

    print(f"\n  AUC счёта «противоречит»:           {auc(lambda s: s['противоречит']):.3f}")
    print(f"  AUC доли противоречий на единицу:   "
          f"{auc(lambda s: s['противоречит']/max(s['единиц'],1)):.3f}")
    print("\n  разделение по порогу (счёт противоречий):")
    for k in (1, 2, 3):
        R = sum(1 for s in поз if s["противоречит"] >= k) / len(поз)
        F = sum(1 for s in отр if s["противоречит"] >= k) / len(отр)
        print(f"    >={k}: полнота {R:.3f}  доля ложных {F:.3f}")
    # Слой коротких ответов — тот, где умер судья-оценщик и где живёт 91% боевого набора.
    for имя, гр in (("короткие (<500 симв)", lambda s: s["ответ_симв"] < 500),
                    ("длинные (>=500)", lambda s: s["ответ_симв"] >= 500)):
        п = [s for s in поз if гр(s)]
        о = [s for s in отр if гр(s)]
        if not п or not о:
            print(f"\n  {имя}: поз={len(п)} отр={len(о)} — мало данных")
            continue
        print(f"\n  {имя}: поз={len(п)} отр={len(о)}")
        for k in (1, 2):
            R = sum(1 for s in п if s["противоречит"] >= k) / len(п)
            F = sum(1 for s in о if s["противоречит"] >= k) / len(о)
            print(f"    >={k}: полнота {R:.3f}  доля ложных {F:.3f}")
    медиана = statistics.median([s["секунд"] for s in строки])
    print(f"\n  медиана времени на ответ: {медиана:.1f} с → "
          f"{медиана*540/60:.0f} мин на 540 строк ПОСЛЕДОВАТЕЛЬНО "
          f"(бюджет 30, значит нужен батчинг или отсечение)")


if __name__ == "__main__":
    raise SystemExit(main())

"""Санити-прогон судьи на выданных 46 примерах.

Что этот прогон МОЖЕТ показать и что не может — разграничено заранее, потому что соблазн
прочитать в нём больше, чем в нём есть, здесь очень велик.

**Не может показать качество.** `[замер §0в]` Выданный сэмпл перевёрнут по стратам
относительно реального набора: в страте с вызовом инструмента у него 21 положительный на
2 отрицательных против реальных 34 на 134, а в текстовой — 2 на 21 против реальных 96 на
276. Поэтому F1 на сэмпле не значит ничего, и правило `has_tool_call`, дававшее здесь 0.913,
на реальном наборе дало 0.2282.

**Может показать три вещи, каждая из которых способна остановить работу до сабмита:**

1. **Работает ли принудительный формат.** Доля неразобранных ответов. `[замер]` На свободном
   формате комплаенс был 0.74, и все отказы приходились на длинные окна.
2. **Не выродилась ли шкала.** Если модель ставит всем одну и ту же оценку, скор перестаёт
   различать что-либо, и порог двигать будет нечего.
3. **Не переусердствовал ли запрет молчать.** Требование «выдай находку всегда» может
   развернуться в обратное смещение: модель начнёт находить нарушения там, где их нет.
   Меряется долей суровых оценок на ОТРИЦАТЕЛЬНЫХ примерах текстовой страты — единственной
   страте, где сэмпл даёт сколько-нибудь пригодный материал (21 пример).

Гейты записаны ЗДЕСЬ, до прогона, чтобы результат мог опровергнуть замысел, а не был
подогнан под него задним числом.

    python scripts/judge_sanity.py --host http://100.81.218.50:11434 --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka import ollama  # noqa: E402
from sverka.judge import parse_verdict, выбрать_порог, предсказать_метрики  # noqa: E402
from sverka.judge_prompt import JSON_SCHEMA, build_prompt  # noqa: E402
from sverka.transcript import parse_response, parse_transcript  # noqa: E402

# ── ГЕЙТЫ. Записаны до прогона. Провал любого означает «чинить, а не сабмитить». ──────────
ГЕЙТЫ = {
    "доля_неразобранных": (0.02, "меньше"),   # принудительный формат обязан работать
    "межквартильный_размах": (0.10, "больше"),  # шкала обязана различать
    "доля_суровых_на_чистых": (0.35, "меньше"),  # запрет молчать не должен стать обратным смещением
}


def очистить(текст: str) -> str:
    """Убрать одиночные суррогаты из вывода модели.

    `[инцидент 30.08.2026]` Прогон упал на 24-м примере из 67: модель выдала обломок эмодзи
    — половину суррогатной пары, — и `json.dumps(..., ensure_ascii=False)` с записью в utf-8
    бросил `UnicodeEncodeError: surrogates not allowed`. Потеряли сорок минут работы уже
    посчитанного. Строки, приходящие от модели, должны чиститься на границе, а не считаться
    валидным текстом просто потому, что это строка Python.
    """
    return (текст or "").encode("utf-8", "surrogatepass").decode("utf-8", "replace")


def спросить(host: str, model: str, system: str, user: str, timeout: int,
             num_ctx: int) -> tuple[str, float, int, bool]:
    """Один запрос к ollama с принудительной схемой ответа.

    `think: false` обязателен: `[инцидент]` у рассуждающих моделей вся квота вывода уходит
    в рассуждение, и поле ответа приходит пустым. Ошибка тихая — выглядит как отказ модели.

    Окно и доставку промпта стережёт `sverka.ollama`: судья, которому дают пятнадцать тысяч
    токенов контекста, при забытом `num_ctx` читал последние две тысячи, и мерился тогда
    не судья, а обрезка `[замер 31.08.2026]`.
    """
    о = ollama.спросить(host, model, system, user, num_ctx=num_ctx, формат=JSON_SCHEMA,
                        # `[замер v3]` при 400 не влезали 4 ответа из 46
                        num_predict=900, думать=False, timeout=timeout)
    return очистить(о.текст), о.секунд, о.дошло, о.вывод_оборван


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://100.81.218.50:11434")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--data", default="data/raw/valid.parquet")
    ap.add_argument("--out", default="data/judge_sanity.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-ctx", type=int, default=32768,
                    help="окно модели; обязано вмещать промпт целиком, иначе ollama режет молча")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    if args.data.endswith(".jsonl"):
        # Корпусная выборка: те же поля, но без pandas и без parquet. Нужна, чтобы мерить
        # долю ложных на НАДЁЖНЫХ отрицательных из текстовой страты — их 24 160, тогда как
        # выданный сэмпл давал 21.
        строки = [json.loads(s) for s in Path(args.data).read_text(encoding="utf-8").splitlines() if s.strip()]
        d = строки[: args.limit] if args.limit else строки
        итератор = lambda: enumerate(d, 1)  # noqa: E731
    else:
        import pandas as pd
        таблица = pd.read_parquet(args.data)
        if args.limit:
            таблица = таблица.head(args.limit)
        d = [dict(r) for _, r in таблица.iterrows()]
        итератор = lambda: enumerate(d, 1)  # noqa: E731

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    сделано = set()
    if out.exists():
        # Прогон на CPU идёт около минуты на пример, поэтому он обязан продолжаться
        # с места обрыва, а не начинаться заново.
        #
        # ⚠️ Но продолжать можно ТОЛЬКО прогон с теми же условиями. `[ревью 31.08.2026]`
        # Прежняя редакция набирала `сделано` по одному лишь `id`, и запуск на файле
        # от 30.08 не делал НИ ОДНОГО обращения к модели, после чего печатал отчёт
        # по отозванным строкам как по свежему замеру — воспроизведено на мёртвом хосте.
        чужие = 0
        for строка in out.read_text(encoding="utf-8").splitlines():
            try:
                z = json.loads(строка)
            except Exception:  # noqa: BLE001
                continue
            свои = (z.get("модель") == args.model and z.get("окно") == args.num_ctx
                    and "дошло_токенов" in z)
            if свои:
                сделано.add(z["id"])
            else:
                чужие += 1
        if чужие:
            print(f"[ОТКАЗ] в {out} лежат {чужие} строк от прогона с другими условиями "
                  f"(другая модель, другое окно или прогон до 31.08.2026). Продолжать нельзя: "
                  f"смешанный файл читается как один замер. Возьмите новый --out.",
                  file=sys.stderr, flush=True)
            return 2
        print(f"[возобновление] уже готово: {len(сделано)}", file=sys.stderr, flush=True)

    with out.open("a", encoding="utf-8") as f:
        for i, r in итератор():
            if r["id"] in сделано:
                continue
            tr = parse_transcript(r["prompt"], strict=False)
            turn = parse_response(r["response"], strict=False)
            п = build_prompt(tr, turn)
            try:
                текст, сек, дошло, оборван = спросить(args.host, args.model, п.system,
                                                      п.user, args.timeout, args.num_ctx)
                ошибка = ""
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                текст, сек, дошло, оборван = "", 0.0, 0, False
                ошибка = f"{type(e).__name__}: {e}"
            except ollama.ВходОбрезан as e:
                # Не роняем прогон целиком: строка помечается недействительной и в отчёте
                # считается отдельно. Потерять сорок минут работы из-за одного длинного
                # промпта — худший размен, чем честно отметить его непригодным.
                текст, сек, дошло, оборван = "", 0.0, 0, False
                ошибка = f"ВходОбрезан: {e}"
            v = parse_verdict(текст)
            f.write(json.dumps({
                "id": r["id"], "label": int(r["label"]),
                "есть_вызов": "TOOL_CALL" in str(r["response"]),
                "домен": r.get("domain", ""),
                "grade": v.grade, "score": v.score, "score_grade": v.score_grade,
                "impact": очистить(v.impact), "parsed": v.parsed,
                "checks": [{k: очистить(str(x)) for k, x in c.items()}
                           for c in v.checks if isinstance(c, dict)],
                "finding": очистить(v.finding)[:300], "evidence": очистить(v.evidence)[:300],
                "усечено": п.truncated, "токенов": п.approx_tokens,
                # Сколько токенов модель прочитала НА САМОМ ДЕЛЕ. Расхождение с «токенов»
                # означает молчаливую обрезку на стороне ollama, а не на нашей.
                "дошло_токенов": дошло,
                # Условия замера — в каждой строке, чтобы дозапись не смешала два прогона
                # и чтобы отчёт нельзя было прочитать в отрыве от того, чем он получен.
                "модель": args.model, "окно": args.num_ctx, "вывод_оборван": оборван,
                "секунд": round(сек, 1), "ошибка": ошибка,
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(d)}] {r['id']} метка={r['label']} оценка={v.grade} "
                  f"скор={v.score} {сек:.0f}с {ошибка}", file=sys.stderr, flush=True)
    отчёт(out)
    return 0


def отчёт(path: Path) -> None:
    строки = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not строки:
        print("пусто"); return
    скоры = [s["score"] for s in строки]
    метки = [s["label"] for s in строки]

    неразобрано = sum(not s["parsed"] for s in строки) / len(строки)
    q = statistics.quantiles(скоры, n=4) if len(скоры) > 3 else [0, 0, 0]
    размах = q[2] - q[0]
    чистые_текстовые = [s for s in строки if s["label"] == 0 and not s["есть_вызов"]]
    суровые = (sum(s["grade"] in ("D", "E") for s in чистые_текстовые) / len(чистые_текстовые)
               if чистые_текстовые else float("nan"))

    print(f"\n{'='*78}\nСАНИТИ-ПРОГОН: {len(строки)} примеров\n{'='*78}")

    # Доставка проверяется ПЕРВОЙ и до всех метрик: отчёт по строкам, где вход не доехал
    # или вывод оборвался, описывает не судью, а канал. `[инцидент 31.08.2026]`
    обрезан_вход = sum("ВходОбрезан" in str(s.get("ошибка", "")) for s in строки)
    оборван = sum(bool(s.get("вывод_оборван")) for s in строки)
    дошли = [s.get("дошло_токенов") or 0 for s in строки if s.get("дошло_токенов")]
    окна = {s.get("окно") for s in строки if s.get("окно")}
    модели = {s.get("модель") for s in строки if s.get("модель")}
    print(f"  условия: модель {' + '.join(sorted(map(str, модели))) or 'не записана'}, "
          f"окно {' + '.join(map(str, sorted(окна))) or 'не записано'}")
    if дошли:
        print(f"  доставка: прочитано токенов медиана {int(statistics.median(дошли))}, "
              f"максимум {max(дошли)}")
    if обрезан_вход or оборван:
        print(f"  ⚠ НЕДЕЙСТВИТЕЛЬНЫХ СТРОК: вход обрезан у {обрезан_вход}, "
              f"вывод оборван у {оборван} — эти строки меряют канал, а не судью")
    факт = {"доля_неразобранных": неразобрано, "межквартильный_размах": размах,
            "доля_суровых_на_чистых": суровые}
    все_ок = True
    for имя, (порог, сторона) in ГЕЙТЫ.items():
        v = факт[имя]
        ок = (v < порог) if сторона == "меньше" else (v > порог)
        все_ок &= ок
        print(f"  {имя:<26} {v:>6.3f}  нужно {сторона} {порог}   {'ок' if ок else 'ПРОВАЛ'}")

    from collections import Counter
    print(f"\n  распределение оценок: {dict(Counter(s['grade'] for s in строки))}")
    печать_страт(строки)

    print(f"\n  ⚠ F1 на сэмпле НЕ считаем: он перевёрнут по стратам относительно реального")
    print(f"    набора. Считаем проекцию по ставкам при реальной доле π = 0.241.")
    for порог in (0.3, 0.5, 0.7, 0.9):
        m = предсказать_метрики(скоры, метки, порог=порог)
        if m:
            print(f"    порог {порог}: полнота {m['полнота']:.3f}, ложных {m['доля_ложных']:.3f} "
                  f"→ прогноз F1 {m['прогноз_F1']:.3f}")
    п = выбрать_порог(скоры, метки)
    if п:
        print(f"\n  порог по maximin на π∈[0.15,0.35]: {п[0]:.2f}, худший F1 {п[1]:.3f}")
    print(f"\n  ИТОГ ГЕЙТОВ: {'можно двигаться к сабмиту' if все_ок else 'ЧИНИТЬ, сабмит не тратить'}")


def печать_страт(строки) -> None:
    for имя, отбор in (("с вызовом", True), ("без вызова", False)):
        s = [x for x in строки if x["есть_вызов"] is отбор]
        поз = [x for x in s if x["label"] == 1]
        отр = [x for x in s if x["label"] == 0]
        tpr = sum(x["score"] >= 0.5 for x in поз) / len(поз) if поз else float("nan")
        fpr = sum(x["score"] >= 0.5 for x in отр) / len(отр) if отр else float("nan")
        print(f"  страта «{имя:<11}» n={len(s):<3} поз={len(поз):<3} отр={len(отр):<3} "
              f"полнота={tpr:.3f} ложных={fpr:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        отчёт(Path(sys.argv[2])); raise SystemExit(0)
    raise SystemExit(main())

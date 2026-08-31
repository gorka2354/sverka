"""Перевод корпуса на русский — так, как устроен боевой набор.

Зачем. `[замер 31.08.2026]` Весь измерительный корпус оказался английским, а боевая
текстовая страта — русской: доля кириллицы выше 0.3 в `nl_free_pos` 0 из 724, в
`text_negatives` 0 из 160, в боевой страте без вызова **23 из 23**. Русская ветвь лексикона
отказа не сработала ни разу на 3173 строках корпуса, английская — ни разу на отрицательных
`valid`. То есть всё, что меряно, меряно не на той половине.

Что переводим и что нет — по замеру структуры боевых данных (медианы доли кириллицы):

| часть | доля кириллицы | действие |
|---|---|---|
| политика в `⟦SYSTEM⟧` | 0.000 | не трогаем |
| схемы инструментов | 0.000 | не трогаем |
| `← TOOL_RESPONSE` (данные БД) | 0.000 | не трогаем |
| реплики `⟦USER⟧` | 0.912 | переводим |
| **проверяемый ответ** | **0.801** | **переводим** |

Проверка сохранности — НАШИМ кодом, а не доверием к модели. Из оригинала и перевода
извлекаются проверяемые единицы (`units.извлечь`), и если множество значений изменилось,
строка помечается: перевод, потерявший число или подменивший идентификатор, испортил бы
ровно тот замер, ради которого делается.

⚠️ Чего этот перевод НЕ даёт. Он не превращает корпус в боевой: остаются другая популяция
(ходы публичных траекторий против набора организаторов) и отсутствие домена telecom.
Он снимает ровно одно из трёх расхождений — языковое.

    python scripts/translate_corpus.py --in data/nl_free_pos.jsonl --n 300 \\
        --out data/nl_free_pos_ru.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka import ollama
from sverka.otvod import тело
from sverka.units import извлечь

СИД = 20260831

SYSTEM = """Ты переводишь реплики службы поддержки с английского на русский для корпуса данных.

ЖЁСТКИЕ ПРАВИЛА:
1. Переводи ТОЛЬКО связный текст. Всё остальное копируй СИМВОЛ В СИМВОЛ.
2. НЕ переводи и НЕ меняй: идентификаторы (HAT123, W2575533, gift_card_6276644), коды
   аэропортов (JFK, ORD), суммы и числа ($226.49, 1,628), даты в формате May 25, 2024,
   названия классов обслуживания (basic economy, business), имена людей.
3. Сохраняй разметку: звёздочки, дефисы списков, переносы строк — как в оригинале.
4. Не исправляй ошибки автора, не сглаживай формулировки, не добавляй вежливости.
   Если в оригинале агент отказывает — отказ обязан остаться таким же прямым.
5. Верни ТОЛЬКО перевод, без пояснений и без кавычек вокруг."""

_МАРКЕР = re.compile(r"^(⟦[^⟧]*⟧)\s*")
# Стандартная фраза симулятора τ²-bench при передаче человеку. В боевых данных её нет
# ни разу (0 из 46), в пуле — 272 из 724. Это артефакт генератора, а не свойство ответов,
# и строки с ним из корпуса исключаются: иначе корпус остаётся непохожим на боевой
# по признаку, который сам же лексикон отказа и ловит.
ШАБЛОН_СИМУЛЯТОРА = re.compile(r"TRANSFERRED TO A HUMAN AGENT", re.IGNORECASE)


def значения(текст: str) -> set[str]:
    return {е.значение for е in извлечь(текст)}


def перевести(host, model, текст, num_ctx, timeout):
    о = ollama.спросить(host, model, SYSTEM, текст, num_ctx=num_ctx, num_predict=1200,
                        температура=0.0, думать=False, timeout=timeout, строго=False)
    return о.текст.strip(), о.секунд


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--in", dest="вход", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    строки = [json.loads(s) for s in
              Path(args.вход).read_text(encoding="utf-8").splitlines() if s.strip()]
    # дедуп по ходу: одна симуляция даёт несколько требований на один и тот же ответ
    видели, уник = set(), []
    for r in строки:
        к = (r.get("источник"), r.get("sim_index"), r.get("turn_idx"))
        if к in видели:
            continue
        видели.add(к)
        if ШАБЛОН_СИМУЛЯТОРА.search(r["response"]):
            continue
        уник.append(r)
    random.Random(СИД).shuffle(уник)
    взято = уник[: args.n]
    print(f"строк {len(строки)} → уникальных ходов без шаблона симулятора {len(уник)} "
          f"→ берём {len(взято)}", file=sys.stderr, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    сделано = set()
    if out.exists():
        for s in out.read_text(encoding="utf-8").splitlines():
            try:
                сделано.add(json.loads(s)["id"])
            except Exception:  # noqa: BLE001
                continue
        print(f"[возобновление] готово {len(сделано)}", file=sys.stderr, flush=True)

    потеряно = 0
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(взято, 1):
            if r["id"] in сделано:
                continue
            исходный = r["response"]
            м = _МАРКЕР.match(исходный)
            голова = м.group(1) if м else ""
            т = тело(исходный)
            try:
                перевод, сек = перевести(args.host, args.model, т, args.num_ctx, args.timeout)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(взято)}] {r['id'][:30]}: ОШИБКА {type(e).__name__}",
                      file=sys.stderr, flush=True)
                continue
            было, стало = значения(т), значения(перевод)
            цел = было <= стало
            потеряно += 0 if цел else 1
            r2 = dict(r)
            r2["response"] = (голова + "\n" + перевод) if голова else перевод
            r2["оригинал"] = исходный
            r2["единицы_целы"] = цел
            r2["потеряно_значений"] = sorted(было - стало)[:10]
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(взято)}] {r['id'][:34]:<34} {len(т):>5}→{len(перевод):<5} симв "
                  f"{'цел' if цел else 'ПОТЕРЯ ' + ','.join(sorted(было - стало)[:3])} {сек:.0f}с",
                  file=sys.stderr, flush=True)
    отчёт(out)
    return 0


def отчёт(path: Path) -> None:
    строки = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not строки:
        print("пусто")
        return
    кир = re.compile(r"[а-яё]", re.IGNORECASE)
    лат = re.compile(r"[a-z]", re.IGNORECASE)

    def доля(t):
        к, л = len(кир.findall(t)), len(лат.findall(t))
        return к / max(к + л, 1)

    д = sorted(доля(тело(s["response"])) for s in строки)
    цел = sum(1 for s in строки if s["единицы_целы"])
    print(f"\n{'='*70}\nПЕРЕВЕДЕНО: {len(строки)}")
    print(f"  доля кириллицы: медиана {д[len(д)//2]:.3f} (в боевых ответах 0.801)")
    print(f"  единицы сохранены: {цел}/{len(строки)} = {цел/len(строки):.3f}")
    if цел < len(строки):
        print("  ⚠ строки с потерей значений помечены полем «единицы_целы»: "
              "их нельзя брать в замер, где проверяется наличие единиц")


if __name__ == "__main__":
    raise SystemExit(main())

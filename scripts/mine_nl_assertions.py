"""Положительные для ТЕКСТОВОЙ страты из `nl_assertions` публичных траекторий.

Зачем. `[замер 31.08.2026]` Полнота судьи в страте «ответ без вызова инструмента» — это
ставка, от которой зависит всё: там лежат 96 из 130 положительных реального набора,
а размеченных примеров у нас **два**. Проекция F1 при полноте 0.26 равна 0.338, при 0.72 —
0.640, и сузить полосу нельзя ничем, кроме разметки.

Откуда берём. В `reward_info` траекторий τ²-bench лежит поле `nl_assertions`: список
требований на естественном языке с флагом `met` и обоснованием от ревьюера бенчмарка.
Майнер корпуса использовал только общий `reward` и этого поля не касался вовсе, хотя это
готовая функция метки, независимая от нашего судьи. Примеры провалов дословно совпадают
с рубрикой ТЗ:

    «Agent communicates that total cost of upcoming flights is $1,628»
    → «The agent communicated the total cost as $708, which is incorrect»

Это выдуманное число в ответе пользователю, то есть ровно та категория, которую мы ловим.

Как локализуем ход. Ассершен относится к симуляции целиком, а метка нужна на ход. Из текста
ассершена и обоснования достаются **якоря** — суммы, идентификаторы, коды рейсов, имена
собственные, — и ищется ход агента, где якорь встречается. Ход берётся, только если якорь
найден **ровно в одном** ходе: неоднозначная привязка молча испортила бы метку.

⚠️ Чем это НЕ является. Провал ассершена не равен галлюцинации по рубрике ТЗ: часть
провалов — это «агент не сделал того, что требовалось» (пропуск), а пропуск галлюцинацией
не считается. Поэтому выход майнера — **кандидаты на ручную вычитку**, а не готовые метки.
Поле `вид_провала` делит их на «сказал неверное» и «не сделал» по формулировке обоснования,
но это подсказка сортировки, а не приговор.

    python scripts/mine_nl_assertions.py --data ~/sverka/tau2/data --out data/nl_candidates.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.tau2 import iter_turns, render_example  # noqa: E402

# ── Якоря: то, по чему ассершен привязывается к ходу ──────────────────────────────────────
# Числа от трёх знаков (суммы, идентификаторы пользователей), коды вида HAT123 / ABC123,
# идентификаторы с подчёркиванием (gift_card_6276644) и слова-имена из двух заглавных частей.
_ЧИСЛО = re.compile(r"\$?\d[\d,]{2,}(?:\.\d+)?")
_КОД = re.compile(r"\b[A-Z][A-Z0-9]{4,}\b")
_СНЕЙК = re.compile(r"\b[a-z]+(?:_[a-z0-9]+){1,}\b")

# Слова, по которым обоснование ревьюера отличает «сказал неверное» от «не сделал».
# Второе по рубрике ТЗ галлюцинацией не является: пропуск — не выдумка.
_НЕ_СДЕЛАЛ = re.compile(
    r"did not (address|mention|provide|inform|communicate|confirm|ask|offer|check)"
    r"|never (mentioned|addressed|informed)|failed to (mention|address|provide|inform)"
    r"|was not (addressed|mentioned|discussed)|no (mention|indication) of",
    re.I)


def якоря(текст: str) -> set[str]:
    из_чисел = {m.group().lstrip("$").replace(",", "") for m in _ЧИСЛО.finditer(текст)}
    return {a for a in (из_чисел | set(_КОД.findall(текст)) | set(_СНЕЙК.findall(текст)))
            if len(a) >= 4}


def _норм(текст: str) -> str:
    """Убирает разделители разрядов и знак валюты, чтобы `$1,628` и `1628` совпадали."""
    return re.sub(r"[$,]", "", текст or "")


def вид_провала(обоснование: str) -> str:
    return "не_сделал" if _НЕ_СДЕЛАЛ.search(обоснование or "") else "сказал_неверное"


def собрать(файл: Path, *, met: bool = False) -> list[dict]:
    """Ходы, к которым привязывается ассершен с заданным исходом.

    `met=False` — провалившиеся требования, кандидаты в положительные.
    `met=True` — ВЫПОЛНЕННЫЕ требования из решённых симуляций. Это негативный контроль
    для ручной разметки, и он обязателен `[инцидент 31.08.2026]`: первая контрольная группа
    строилась делением провалов на «сказал неверное» и «не сделал» по формулировке
    обоснования, и деление оказалось неверным — бенчмарк пишет «did not communicate X»
    и там, где агент вместо X сообщил ошибочное Y, то есть в случае прямой галлюцинации.
    Разметчики это разглядели, а ключ — нет. Честный контроль может быть только один:
    ход, где требование ВЫПОЛНЕНО и симуляция решена.
    """
    данные = json.loads(файл.read_text(encoding="utf-8"))
    провалы: dict[int, list[dict]] = {}
    for i, s in enumerate(данные.get("simulations") or []):
        ri = s.get("reward_info") or {}
        ассершены = ri.get("nl_assertions") or []
        if met:
            # Контроль обязан быть чистым по всем признакам разом: задача решена, база
            # сошлась, ни одного проваленного требования.
            if float(ri.get("reward") or 0) != 1.0:
                continue
            if any(a.get("met") is False for a in ассершены):
                continue
            плохие = [a for a in ассершены if a.get("met") is True]
        else:
            плохие = [a for a in ассершены if a.get("met") is False]
        if плохие:
            провалы[i] = плохие
    if not провалы:
        return []

    # Ходы группируются по симуляции за ОДИН проход файла: iter_turns читает и парсит его
    # целиком, и звать её на каждый ассершен значило бы перечитывать сотни мегабайт.
    ходы: dict[int, list] = collections.defaultdict(list)
    for tt in iter_turns(файл, only_reward=None, require_tool_calls=False):
        if tt.sim_index in провалы:
            ходы[tt.sim_index].append(tt)

    карточки = []
    for sim_index, плохие in провалы.items():
        сим_ходы = ходы.get(sim_index) or []
        for a in плохие:
            текст = f"{a.get('nl_assertion') or ''} {a.get('justification') or ''}"
            як = якоря(текст)
            if not як:
                continue
            найдено = [tt for tt in сим_ходы
                       if any(я in _норм(tt.turn.text) for я in як)]
            if len(найдено) != 1:
                continue
            tt = найдено[0]
            if tt.turn.tool_calls:
                continue  # это страта «с вызовом», её потолок 0.2615 и она не наше узкое место
            prompt, response = render_example(tt)
            карточки.append({
                "id": f"{tt.domain}__{tt.task_id}::t{tt.turn_idx}#nl{len(карточки)}",
                "prompt": prompt,
                "response": response,
                "домен": tt.domain,
                "reward": tt.reward,
                "ассершен": a.get("nl_assertion") or "",
                "обоснование": a.get("justification") or "",
                "вид_провала": вид_провала(a.get("justification") or ""),
                "met": bool(a.get("met")),
                "якорей": len(як),
                "источник": файл.name,
                "sim_index": sim_index,
                "turn_idx": tt.turn_idx,
            })
    return карточки


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/sverka/tau2/data")
    ap.add_argument("--out", default="data/nl_candidates.jsonl")
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--met", action="store_true",
                    help="собрать ВЫПОЛНЕННЫЕ требования решённых симуляций — негативный контроль")
    args = ap.parse_args()

    файлы = sorted(Path(args.data).expanduser().glob("*.json"))
    if args.limit_files:
        файлы = файлы[: args.limit_files]
    print(f"файлов: {len(файлы)}", file=sys.stderr, flush=True)

    всего: list[dict] = []
    for i, f in enumerate(файлы, 1):
        try:
            к = собрать(f, met=args.met)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(файлы)}] {f.name}: ОШИБКА {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue
        всего += к
        print(f"[{i}/{len(файлы)}] {f.name}: +{len(к)} (всего {len(всего)})",
              file=sys.stderr, flush=True)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for к in всего:
            fh.write(json.dumps(к, ensure_ascii=False) + "\n")

    по_видам = collections.Counter(к["вид_провала"] for к in всего)
    по_доменам = collections.Counter(к["домен"] for к in всего)
    print(f"\n{'='*70}\nКАНДИДАТОВ: {len(всего)} → {out}")
    print(f"  по виду провала: {dict(по_видам)}")
    print(f"  по доменам:      {dict(по_доменам)}")
    print("  ⚠ это КАНДИДАТЫ: провал ассершена не равен галлюцинации по рубрике ТЗ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

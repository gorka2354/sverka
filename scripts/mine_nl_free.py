"""Анкер-НЕЗАВИСИМЫЙ пул положительных: привязка требования к ходу по ПОЗИЦИИ.

Зачем понадобился второй майнер. `[замер 31.08.2026]` `mine_nl_assertions.py` привязывает
провалившееся требование к ходу по вхождению **якоря** — числа от трёх знаков, кода
в верхнем регистре, snake_case-идентификатора. Это ровно тот признак, который достаёт
`units.извлечь`, и следствие измерено:

| набор | доля ответов БЕЗ извлекаемых единиц |
|---|---|
| пул `mine_nl_assertions` | 0 / 213 = 0.000 |
| боевые отрицательные текстовой страты | 15 / 21 = 0.714 |
| `text_negatives` | 103 / 160 = 0.644 |

То есть в том пуле у каждого положительного есть числа **по построению**, а в боевых данных
их нет у двух третей ответов. Любая величина, зависящая от наличия единиц, на нём считается
тавтологично. А решающая величина инкремента — ровно такая: r1, доля положительных, которые
проходят отвод (нет вызова, нет единиц, нет маркера невозможности).

Как привязываем здесь. Ход выбирается по **позиции в диалоге**, признаком, ортогональным
содержанию ответа: последний текстовый ход агента после последнего вызова инструмента,
и только если такой ход в симуляции **единственный**. Требование единственности — защита
от тихой ошибки: при нескольких кандидатах непонятно, к какому из них относится вердикт
ревьюера, и метка была бы поставлена наугад.

Контроль. Прогон с `--met` берёт ВЫПОЛНЕННЫЕ требования решённых симуляций той же
процедурой. Если контроль ведёт себя как положительные, значит привязка по позиции ловит
не то, и пул выбрасывается целиком, а не чинится задним числом.

    python3 scripts/mine_nl_free.py --data ~/sverka/tau2/data --out data/nl_free_pos.jsonl
    python3 scripts/mine_nl_free.py --data ~/sverka/tau2/data --met --out data/nl_free_ctl.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Обоснование ревьюера отличает «сказал неверное» от «не сделал». Второе по рубрике ТЗ
# галлюцинацией не является. ⚠️ Деление грубое и уже один раз ошиблось: формулировка
# «did not communicate X» стоит и там, где агент вместо X сообщил ошибочное Y. Поэтому
# вид пишется в строку колонкой, а не используется как фильтр.
import re

from sverka.tau2 import iter_turns, render_example

_НЕ_СДЕЛАЛ = re.compile(
    r"did not (address|mention|provide|inform|communicate|confirm|ask|offer|check)"
    r"|never (mentioned|addressed|informed)|failed to (mention|address|provide|inform)"
    r"|was not (addressed|mentioned|discussed)|no (mention|indication) of",
    re.IGNORECASE)


def собрать(файл: Path, *, met: bool = False) -> list[dict]:
    данные = json.loads(файл.read_text(encoding="utf-8"))
    нужные: dict[int, list[dict]] = {}
    for i, s in enumerate(данные.get("simulations") or []):
        ri = s.get("reward_info") or {}
        ассершены = ri.get("nl_assertions") or []
        if met:
            if float(ri.get("reward") or 0) != 1.0:
                continue
            if any(a.get("met") is False for a in ассершены):
                continue
            взять = [a for a in ассершены if a.get("met") is True]
        else:
            взять = [a for a in ассершены if a.get("met") is False]
        if взять:
            нужные[i] = взять
    if not нужные:
        return []

    # Файл читается один раз: iter_turns парсит его целиком, и звать её на симуляцию
    # значило бы перечитывать сотни мегабайт.
    ходы: dict[int, list] = collections.defaultdict(list)
    for tt in iter_turns(файл, only_reward=None, require_tool_calls=False):
        if tt.sim_index in нужные:
            ходы[tt.sim_index].append(tt)

    карточки = []
    for sim_index, взять in нужные.items():
        сим = sorted(ходы.get(sim_index) or [], key=lambda t: t.turn_idx)
        if not сим:
            continue
        # Позиция последнего вызова инструмента в диалоге. Всё, что после него, — это
        # то, чем агент отчитался пользователю, и туда же смотрит вердикт ревьюера.
        последний_вызов = max((t.turn_idx for t in сим if t.turn.tool_calls), default=-1)
        хвост = [t for t in сим
                 if t.turn_idx > последний_вызов and not t.turn.tool_calls
                 and (t.turn.text or "").strip()]
        if len(хвост) != 1:
            continue  # неоднозначно — метка была бы поставлена наугад
        tt = хвост[0]
        prompt, response = render_example(tt)
        for a in взять:
            карточки.append({
                "id": f"{tt.domain}__{tt.task_id}::t{tt.turn_idx}#free{len(карточки)}",
                "prompt": prompt,
                "response": response,
                "домен": tt.domain,
                "reward": tt.reward,
                "требование": a.get("nl_assertion") or "",
                "обоснование": a.get("justification") or "",
                "вид": ("не_сделал" if _НЕ_СДЕЛАЛ.search(a.get("justification") or "")
                        else "сказал_неверное"),
                "met": bool(a.get("met")),
                "ходов_в_хвосте": len(хвост),
                "источник": файл.name,
                "sim_index": sim_index,
                "turn_idx": tt.turn_idx,
            })
    return карточки


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/sverka/tau2/data")
    ap.add_argument("--out", default="data/nl_free_pos.jsonl")
    ap.add_argument("--met", action="store_true",
                    help="выполненные требования решённых симуляций — негативный контроль")
    args = ap.parse_args()

    файлы = sorted(Path(args.data).expanduser().glob("*.json"))
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

    print(f"\n{'='*70}\nСТРОК: {len(всего)} → {out}")
    print(f"  по виду:    {dict(collections.Counter(к['вид'] for к in всего))}")
    print(f"  по доменам: {dict(collections.Counter(к['домен'] for к in всего))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

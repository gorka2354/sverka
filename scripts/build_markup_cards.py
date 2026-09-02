"""Карточки для ручной вычитки: 50 штук, слепая смесь, готовые к разметке за вечер.

Зачем именно ручная вычитка. `[замер 31.08.2026]` Всё упирается в одну ставку — полноту
судьи в страте «ответ без вызова инструмента». Проекция F1 при полноте 0.26 равна 0.338,
при 0.72 — 0.640; порог, за которым мы обходим лидера, равен 0.470. Разметить эту ставку
нечем, кроме глаз: положительных примеров в этой страте у нас два.

Что делает скрипт. Берёт кандидатов из `mine_nl_assertions.py` и собирает карточку:
ответ агента, срез контекста, где лежит проверяемое, и вердикт ревьюера бенчмарка
(ассершен + обоснование) как независимую подсказку.

⚠️ **Слепая контрольная группа — обязательна.** Кандидаты добыты фильтром «провалившийся
nl_assertion», и если размечать только их, разметка опишет фильтр, а не положительные:
разметчик, которому все пятьдесят карточек показывают с готовым обвинением, поставит
пятьдесят «да». Поэтому в набор подмешиваются карточки вида `не_сделал` — те же провалы
ассершена, но по рубрике ТЗ это пропуск, а не галлюцинация. Внешне они неотличимы,
вид провала в карточку НЕ попадает, порядок перемешан фиксированным сидом. Если разметка
осмысленна, «да» ляжет преимущественно на первую группу; если «да» окажется везде —
разметку придётся выбросить, и лучше узнать это по контрольной группе, чем по сабмиту.

    python scripts/build_markup_cards.py --n 50 --out data/markup_cards.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.slices import отобрать  # noqa: E402
from sverka.transcript import parse_response, parse_transcript  # noqa: E402
from sverka.units import извлечь  # noqa: E402

СИД = 20260831


def срез_контекста(prompt: str, response: str, предел: int = 6000) -> str:
    """Куски контекста, где может лежать проверяемое в ответе.

    Тот же слепой отбор, что идёт судье: карточка обязана показывать разметчику то же,
    что видит машина, иначе разметка будет мерить не ту задачу.
    """
    turn = parse_response(response, strict=False)
    единицы = извлечь(turn.text or response)
    куски: list[str] = []
    видели: set[str] = set()
    for ед in единицы[:6]:
        срез = отобрать(prompt, ед)
        if срез.пуст:
            continue
        т = срез.текст()
        if т[:200] in видели:
            continue
        видели.add(т[:200])
        куски.append(т)
        if sum(len(k) for k in куски) > предел:
            break
    если_есть = "\n\n[…]\n\n".join(куски)[:предел]
    if если_есть:
        return если_есть
    # Запасной срез для лёгкого хода. `[замер 31.08.2026]` Отбор идёт по якорям ОТВЕТА,
    # а у ответа с нулём чисел и идентификаторов якорей нет — срез выходит пустым, и
    # карточка показывает разметчику обвинение без материала. Между тем именно лёгкие
    # ходы решают исход: их 91% в боевых данных, и там судья слеп `[§0е]`.
    # Поэтому берётся то, по чему судят о ложном отказе и о нарушении политики: начало
    # системного сообщения (правила) и хвост диалога (что происходило прямо перед ходом).
    tr = parse_transcript(prompt, strict=False)
    политика = next((t.text for t in tr.turns if t.role == "SYSTEM" and t.text), "")
    хвост = "\n\n".join(f"[{t.role}] {(t.text or '').strip()}"
                        for t in tr.turns[-6:] if (t.text or "").strip())
    голова = политика.strip()[: предел // 2]
    return (f"{голова}\n\n[…]\n\n{хвост.strip()[: предел // 2]}").strip()[:предел]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="вход", default="data/nl_candidates.jsonl")
    ap.add_argument("--чистые", default="data/nl_clean.jsonl",
                    help="выполненные требования решённых симуляций — негативный контроль")
    ap.add_argument("--out", default="data/markup_cards.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--контроль", type=int, default=10,
                    help="сколько карточек взять из вида «не_сделал» — слепой контроль")
    ap.add_argument("--только-лёгкие", type=int, default=0, metavar="ЕДИНИЦ",
                    help="брать только ходы, где проверяемых единиц не больше этого числа. "
                         "Слой лёгких ходов решает исход (91% боевых), а размечено в нём "
                         "восемь положительных — на восьми точках полнота не измеряется")
    ap.add_argument("--исключить", default="", metavar="JSONL",
                    help="файл с уже размеченными строками: их id в набор не попадут")
    args = ap.parse_args()

    строки = [json.loads(s) for s in
              Path(args.вход).read_text(encoding="utf-8").splitlines() if s.strip()]
    чистые = [json.loads(s) for s in
              Path(args.чистые).read_text(encoding="utf-8").splitlines() if s.strip()]

    если_исключать: set[str] = set()
    if args.исключить:
        # Сравнение по ХОДУ, а не по полному id. `[промах 31.08.2026]` В первой редакции
        # сравнивались строки целиком, и восемь карточек «новой» партии оказались другими
        # требованиями к уже размеченным ходам: `...::t11#nl11` не равно `...::t11#nl12`,
        # а ход у них один и тот же, и разметчик размечал заново то, что уже размечено.
        если_исключать = {json.loads(s)["id"].split("#")[0] for s in
                          Path(args.исключить).read_text(encoding="utf-8").splitlines()
                          if s.strip()}
        строки = [r for r in строки if r["id"].split("#")[0] not in если_исключать]
        чистые = [r for r in чистые if r["id"].split("#")[0] not in если_исключать]

    if args.только_лёгкие:
        # Тяжесть меряется по ответу — тем же счётчиком единиц, каким считается разрез
        # в отчёте судьи. Иначе «лёгкие» в наборе и «лёгкие» в отчёте значили бы разное.
        лёгок = lambda r: len(извлечь(  # noqa: E731
            (parse_response(r["response"], strict=False).text or r["response"]))) <= args.только_лёгкие
        строки = [r for r in строки if лёгок(r)]
        чистые = [r for r in чистые if лёгок(r)]

    rnd = random.Random(СИД)
    rnd.shuffle(строки)
    rnd.shuffle(чистые)
    взято = строки[: args.n - args.контроль] + чистые[: args.контроль]
    rnd.shuffle(взято)
    if len(взято) < args.n:
        print(f"⚠ набралось {len(взято)} карточек вместо {args.n}: "
              f"кандидатов {len(строки)}, чистых {len(чистые)} после отбора")

    карточки = []
    for i, r in enumerate(взято, 1):
        tr = parse_transcript(r["prompt"], strict=False)
        turn = parse_response(r["response"], strict=False)
        запрос = ""
        for t in reversed(tr.turns):
            if t.role == "USER" and (t.text or "").strip():
                запрос = t.text.strip()[:1200]
                break
        карточки.append({
            "n": i,
            "id": r["id"],
            "домен": r["домен"],
            "запрос": запрос,
            "ответ": (turn.text or r["response"]).strip(),
            "срез": срез_контекста(r["prompt"], r["response"]),
            # Два майнера называют одно и то же по-разному: `mine_nl_assertions` кладёт
            # «ассершен», `mine_nl_free` — «требование». Поле одно и то же: формулировка
            # требования из разметки бенчмарка.
            "требование": r.get("ассершен") or r.get("требование", ""),
            # Обоснование ревьюера в карточку НЕ идёт: у выполненного требования оно
            # начинается с «The agent did ...», и контроль выдал бы себя первой строкой.
            # Флаг met — тем более: он и есть предмет слепой проверки.
            "_met": bool(r.get("met")),
        })

    out = Path(args.out)
    out.write_text(json.dumps(карточки, ensure_ascii=False, indent=1), encoding="utf-8")
    ключ = out.with_name(out.stem + "_ключ.json")
    ключ.write_text(json.dumps({к["n"]: ("чистая" if к["_met"] else "кандидат")
                                for к in карточки}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    for к in карточки:
        к.pop("_met", None)
    out.write_text(json.dumps(карточки, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"карточек: {len(карточки)} → {out}  ({out.stat().st_size // 1024} КБ)")
    print(f"ключ контрольной группы (не открывать до разметки): {ключ}")
    print(f"  медиана среза: {sorted(len(к['срез']) for к in карточки)[len(карточки)//2]} символов")
    print(f"  пустых срезов: {sum(1 for к in карточки if not к['срез'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

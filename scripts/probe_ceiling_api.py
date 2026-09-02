"""Потолок задачи: решается ли она чтением контекста ВООБЩЕ.

Зачем. Пять линий опровергнуты, и все пять замеров сделаны на `qwen3.5:9b` в четырёх битах —
худшей конфигурации по каждой оси разом. Неснятым остаётся один конфаунд: **класс модели**.
Если сильная модель на тех же 46 примерах с настоящими метками даёт AUC около 0.85, значит
задача решается чтением контекста, и вкладываться надо в модель. Если около 0.55 — линия
закрыта окончательно, и слоты идут на удержание пола.

⚠️ **Результат НЕ едет в решение.** Правила соревнования запрещают проприетарные технологии,
требующие возмездного приобретения прав, а архив уезжает на изолированную машину. Это
измерительный инструмент, как и машинная разметка корпуса: данные готовим чем угодно,
в сабмит едет только наш код.

⚠️ **Жёсткий потолок трат.** Ключ рабочий, и лимит здесь не рекомендация: перед КАЖДЫМ
запросом считается его цена по длине промпта, и если она выводит сумму за предел — запрос
не отправляется вовсе. Останов до превышения, а не после.

    python scripts/probe_ceiling_api.py --limit-usd 2.0 --model claude-sonnet-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.judge import parse_verdict  # noqa: E402
from sverka.judge_prompt import build_prompt  # noqa: E402
from sverka.transcript import parse_response, parse_transcript  # noqa: E402

# Цены за миллион токенов. Взяты консервативно — по прайсу ПОСЛЕ промо, чтобы оценка
# не оказалась ниже факта. Реальные траты считаются по `usage` из ответа, а эти числа
# нужны только для решения «хватит ли остатка на следующий запрос».
ЦЕНЫ = {  # $ за миллион токенов, вход/выход
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

КЛЮЧ_ИЗ = Path(r"C:\Users\pesto\Desktop\Email Brain\email-brain\.env")


class ЗапрещеноВладельцем(SystemExit):
    """Ключ рабочий, и владелец запретил им пользоваться."""


def проверить_разрешение(разрешено: bool) -> None:
    """⛔ Стоп-кран, поставленный по прямому указанию 01.09.2026.

    Ключ, который читает `ключ()`, — **рабочий**, он используется в другом проекте
    и оплачивается не из личных средств. Владелец сказал дословно: «соннет прошу больше
    не тестить, это апи не моё а рабочее, больше не трать и не трогай его».

    Замер потолка уже сделан и повторять его незачем: Sonnet 5 на 115 размеченных ходах
    текстовой страты дала AUC 0.698, полноту 0.462 при доле ложных 0.175, проекция по
    стратам F1 = 0.556 `[inc-4, замер 7, $5.79]`. Результат записан, вопрос закрыт.

    Поэтому запуск требует явного флага `--я-подтверждаю-трату-рабочего-ключа`. Флаг
    существует не для удобства, а чтобы случайный запуск скрипта — из истории команд,
    из документации, из чужой сессии — упирался в отказ, а не в счёт.
    """
    if not разрешено:
        raise ЗапрещеноВладельцем(
            "ОТКАЗ: ключ рабочий, владелец запретил траты 01.09.2026. Потолок задачи уже "
            "измерен (Sonnet 5: AUC 0.698, F1 по стратам 0.556, см. inc-4 замер 7). "
            "Если запуск действительно нужен — спросить владельца и передать флаг "
            "--я-подтверждаю-трату-рабочего-ключа."
        )


def ключ() -> str:
    """Ключ из окружения или из .env соседнего проекта. НИКОГДА не печатается."""
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    if КЛЮЧ_ИЗ.exists():
        for строка in КЛЮЧ_ИЗ.read_text(encoding="utf-8", errors="replace").splitlines():
            if строка.strip().startswith("ANTHROPIC_API_KEY="):
                return строка.partition("=")[2].strip().strip('"').strip("'")
    raise SystemExit("нет ANTHROPIC_API_KEY ни в окружении, ни в .env Email Brain")


def спросить(k, model, system, user, timeout=180, max_tokens=1200):
    тело = json.dumps({
        "model": model,
        # `[инцидент 01.09.2026]` При 1200 оборвались 10 ответов из 115 на размеченном
        # корпусе, и семь из них были положительными: разборщик отдал их как «модель
        # не ответила» со скором 0.0, то есть обрыв бил прямо по полноте. AUC при этом
        # читалась как 0.636 вместо 0.718 на разобранных. Лимит вывода — такой же тихий
        # источник вранья, как окно у ollama: ответ приходит, просто он неполный.
        "max_tokens": max_tokens,
        # ⚠️ Рассуждение выключено намеренно. `[замер 31.08.2026]` С включённым (у Sonnet 5
        # это режим по умолчанию) модель израсходовала на него все 900 токенов вывода
        # и вернула ОДИН блок `thinking` с пустым текстом: `stop_reason: max_tokens`,
        # ответа нет. Это та же ловушка, что записана про ollama, только в другом API.
        # Нам нужен вердикт, а не рассуждение, и цена вывода здесь втрое выше цены входа.
        "thinking": {"type": "disabled"},
        # temperature не задаётся: у новых моделей параметр объявлен устаревшим и запрос
        # с ним отклоняется целиком `[замер 31.08.2026: HTTP 400 temperature is deprecated]`.
        # Политика и схемы инструментов повторяются от примера к примеру внутри домена,
        # поэтому системная часть помечается кешируемой: чтение кеша дешевле входа
        # примерно вдесятеро и заметно раздвигает потолок в пределах того же лимита.
        "system": [{"type": "text", "text": system,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=тело,
        headers={"content-type": "application/json", "x-api-key": k,
                 "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        о = json.loads(r.read().decode("utf-8"))
    текст = "".join(б.get("text", "") for б in о.get("content", []) if б.get("type") == "text")
    u = о.get("usage", {})
    return текст, u, time.time() - t0


def цена(u: dict, model: str) -> float:
    вх, вых = ЦЕНЫ.get(model, (2.0, 10.0))
    обычный = u.get("input_tokens", 0)
    запись = u.get("cache_creation_input_tokens", 0)
    чтение = u.get("cache_read_input_tokens", 0)
    # Запись в кеш дороже входа на четверть, чтение — вдесятеро дешевле.
    return (обычный * вх + запись * вх * 1.25 + чтение * вх * 0.1 +
            u.get("output_tokens", 0) * вых) / 1_000_000


def auc(строки) -> float:
    p = [s["score"] for s in строки if s["label"] == 1]
    n = [s["score"] for s in строки if s["label"] == 0]
    if not p or not n:
        return float("nan")
    w = sum((1 if a > b else 0.5 if a == b else 0) for a in p for b in n)
    return w / (len(p) * len(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--limit-usd", type=float, default=2.0)
    ap.add_argument("--data", default="data/raw/valid.parquet")
    ap.add_argument("--out", default="data/ceiling_api.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--я-подтверждаю-трату-рабочего-ключа", action="store_true",
                    dest="разрешено",
                    help="ключ рабочий и не принадлежит владельцу проекта; без этого флага "
                         "скрипт не делает НИ ОДНОГО запроса")
    ap.add_argument("--max-tokens", type=int, default=1200,
                    help="потолок вывода. При 1200 оборвались 10 ответов из 115, семь "
                         "из них положительные — обрыв читается как «не ответила»")
    ap.add_argument("--budget-chars", type=int, default=60_000)
    ap.add_argument("--спасать-якоря", action="store_true",
                    help="та же подача, что в лучшем прогоне 27B — иначе сравниваются "
                         "две вещи разом")
    args = ap.parse_args()

    проверить_разрешение(args.разрешено)
    k = ключ()
    if args.data.endswith(".jsonl"):
        # Размеченный корпус текстовой страты: 115 ходов против двух положительных,
        # которые даёт по этой страте выданный сэмпл `[inc-4]`. Ради него и добавлена
        # эта ветка — на сэмпле «потолок» меряется там, где боевых данных почти нет.
        записи = [json.loads(s) for s in
                  Path(args.data).read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        import pandas as pd
        записи = [dict(r) for _, r in pd.read_parquet(args.data).iterrows()]
    # Чередование классов: лимит трат может оборвать прогон на любой строке, и при
    # исходном порядке обрыв оставил бы выборку перекошенной, а AUC — несчитаемой.
    поз = [r for r in записи if r["label"] == 1]
    отр = [r for r in записи if r["label"] == 0]
    порядок = [x for пара in zip(поз + [None] * len(отр), отр + [None] * len(поз))
               for x in пара if x is not None]
    if args.limit:
        порядок = порядок[: args.limit]
    d = порядок

    out = Path(args.out)
    сделано = {}
    if out.exists():
        for s in out.read_text(encoding="utf-8").splitlines():
            try:
                z = json.loads(s)
                if z.get("модель") == args.model:
                    сделано[z["id"]] = z
            except Exception:  # noqa: BLE001
                continue
        print(f"[возобновление] готово {len(сделано)}", file=sys.stderr, flush=True)

    потрачено = sum(z.get("цена", 0.0) for z in сделано.values())
    вх_цена = ЦЕНЫ.get(args.model, (2.0, 10.0))[0]
    строки = list(сделано.values())

    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(d, 1):
            if r["id"] in сделано:
                continue
            tr = parse_transcript(str(r["prompt"]), strict=False)
            turn = parse_response(str(r["response"]), strict=False)
            # Подача обязана совпадать с прогоном, с которым сравниваем, иначе замерятся
            # две вещи разом. `[замер 01.09.2026]` У 27B отбор по якорям поднял AUC сэмпла
            # 0.667 → 0.747, так что разница в подаче тут не косметическая.
            п = build_prompt(tr, turn, budget_chars=args.budget_chars,
                             спасать_якоря=args.спасать_якоря)
            # ОЦЕНКА ДО ОТПРАВКИ: 3.5 символа на токен — нижняя граница замеренного
            # диапазона 3.01–4.26, то есть оценка токенов ЗАВЫШЕНА, и лимит не будет
            # пройден по недосмотру.
            # Оценка с запасом: 3.0 симв/токен (нижняя граница замеренного диапазона)
            # плюс полный max_tokens по цене вывода. Завышаем намеренно — лимит должен
            # срабатывать раньше, а не позже.
            вых_цена = ЦЕНЫ.get(args.model, (2.0, 10.0))[1]
            оценка = ((len(п.system) + len(п.user)) / 3.0 * вх_цена + 1200 * вых_цена) / 1_000_000
            if потрачено + оценка > args.limit_usd:
                print(f"[СТОП] следующий запрос стоил бы ~${оценка:.3f}, "
                      f"потрачено ${потрачено:.3f} из ${args.limit_usd} — "
                      f"останавливаемся ДО превышения", file=sys.stderr, flush=True)
                break
            try:
                текст, u, сек = спросить(k, args.model, п.system, п.user,
                                        max_tokens=args.max_tokens)
            except urllib.error.HTTPError as e:
                тело = e.read().decode("utf-8", "replace")[:300]
                print(f"[{i}] HTTP {e.code}: {тело}", file=sys.stderr, flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{i}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue
            c = цена(u, args.model)
            потрачено += c
            v = parse_verdict(текст)
            з = {"id": r["id"], "label": int(r["label"]), "grade": v.grade,
                 "score": v.score, "parsed": v.parsed, "модель": args.model,
                 "цена": round(c, 5), "секунд": round(сек, 1),
                 "usage": u, "есть_вызов": "TOOL_CALL" in str(r["response"])}
            строки.append(з)
            f.write(json.dumps(з, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(d)}] {r['id'][:34]:<34} метка={r['label']} "
                  f"оценка={v.grade} скор={v.score} ${c:.4f} итого ${потрачено:.3f} {сек:.0f}с",
                  file=sys.stderr, flush=True)

    print(f"\n{'='*72}\nПОТОЛОК ЗАДАЧИ: {args.model}, примеров {len(строки)}, "
          f"потрачено ${потрачено:.3f} из ${args.limit_usd}\n{'='*72}")
    if len(строки) < 4:
        print("  мало данных")
        return 0
    a = auc(строки)
    print(f"  AUC = {a:.3f}   (локальная qwen3.5:9b Q4 давала 0.731 на тех же 46)")
    неразобрано = sum(1 for s in строки if not s["parsed"]) / len(строки)
    print(f"  доля неразобранных ответов: {неразобрано:.3f}")
    for порог in (0.3, 0.5, 0.7):
        поз = [s for s in строки if s["label"] == 1]
        отр = [s for s in строки if s["label"] == 0]
        if not поз or not отр:
            continue
        R = sum(1 for s in поз if s["score"] >= порог) / len(поз)
        F = sum(1 for s in отр if s["score"] >= порог) / len(отр)
        # Проекция на боевые страты: 34/134 с вызовом, 96/276 без.
        TP, FP = 130 * R, 410 * F
        P = TP / (TP + FP) if TP + FP else 0
        F1 = 2 * P * R / (P + R) if P + R else 0
        print(f"  порог {порог}: полнота {R:.3f} ложных {F:.3f} → проекция F1 {F1:.3f}")
    print(f"\n  ориентиры: пол 0.3835 | baseline 0.4896 | лидер 0.4918")
    print(f"  ⚠ выборка 50/50 и не репрезентативна; смотреть на AUC, не на F1 сэмпла")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

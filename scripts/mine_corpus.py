"""Сборка корпуса из публичных траекторий τ²-bench в формате организаторов.

Зачем нужен собственный корпус. На выданных 46 примерах ширина доверительного интервала F1
равна 0.253, то есть 0.70 и 0.85 неразличимы. Хуже того, общая метрика там меряет в основном
наличие вызова инструмента: в страте «ответ с вызовом» доля позитива 0.91, в страте
«без вызова» — 0.09. Пока валидация такая, улучшения нечем измерять.

## Откуда берутся метки и чего эта разметка НЕ значит

Разметки на уровне хода в публичных траекториях нет: поле `review` у Sierra всегда пустое.
Поэтому метки здесь **слабые**, выведены из двух независимых сигналов, и это записано
в самих данных, чтобы никто потом не принял их за эталон:

- `weak_negative` — ход из решённой задачи (`reward == 1.0`), на котором не сработал ни один
  детерминированный гейт. Задача доведена до конца, конечное состояние базы совпало
  с эталоном, гейты молчат. Такой ход почти наверняка корректен.
- `weak_positive` — ход из проваленной задачи (`reward == 0.0`), на котором сработал гейт
  с измеренным обогащением. `[замер]` На провалах гейт «недоступный инструмент» срабатывает
  втрое чаще, чем на решённых (1.66% против 0.53%).

Границы, названные прямо. Ход решённой задачи мог нарушить процедурное правило, не изменив
итогового состояния базы, — такой попадёт в негативы ошибочно. Ход проваленной задачи мог
быть корректным, а провалился следующий, — такой попадёт в позитивы ошибочно. Поэтому
корпус годится для замера ОТНОСИТЕЛЬНЫХ изменений и для обучения, но не как эталон
абсолютного качества. Каждая строка несёт `label_source`, чтобы это было видно.

Гейт `M_multi_mutating` в источник позитивов не входит: `[замер]` обогащения на провалах
у него нет вовсе (0.62% против 0.83%), то есть он ловит сложность хода, а не ошибку.

## Что важно в формате

Строки рендерятся ровно в формат организаторов и проходят через тот же парсер, что и боевые
данные. Круговая проверка (рендер → разбор → те же вызовы, схемы и вердикт гейтов) прогнана
на реальных траекториях: 81 из 81.

## Что записывается

Индекс, а не текст: ссылка на файл, номер симуляции и номер хода, метка, источник метки,
сработавшие гейты, домен, reward и длины. Текст восстанавливается детерминированно
функцией `sverka.tau2.render_by_ref`. Первая версия писала полный промпт в каждую строку
и дала файл на 9.9 ГБ при 175 972 строках — непригодный ни к чтению, ни к пересылке.

⚠️ **Независимых групп мало.** `[замер]` 175 972 строки приходятся всего на 375 различных
задач: одна задача повторяется по трайлам и по моделям-участникам. Валидационный сплит
делить по ЗАДАЧЕ, а не по строке и не по файлу, иначе одна и та же задача попадёт
и в обучение, и в проверку.

    python scripts/mine_corpus.py --data ~/sverka/tau2/data --out corpus_index.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.gates import apply_gates, load_known_user_tools  # noqa: E402
from sverka.tau2 import iter_turns, render_example  # noqa: E402

# Гейты, чьё срабатывание на проваленной задаче считается признаком позитива.
# Отбор по измеренному обогащению, а не по интуиции.
POSITIVE_GATES = ("A_unavailable_tool", "B_repeat_failed_call", "C_placeholder_arg",
                  "D_schema_violation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-files", type=int, default=8)
    ap.add_argument("--max-mb", type=float, default=60.0)
    ap.add_argument("--max-sims", type=int, default=None)
    ap.add_argument("--text-turns", action="store_true",
                    help="включать ходы БЕЗ вызова инструмента. `[замер]` Их 176 099 из "
                         "479 837 (36.7%) — и это половина целевой задачи: в выданном сэмпле "
                         "23 из 46 примеров текстовые, а доля позитива там 0.09 против 0.91 "
                         "в страте с вызовами. Детерминированные гейты туда не достают "
                         "по построению, поэтому позитивов оттуда не будет: только негативы "
                         "из решённых задач и кандидаты из проваленных")
    ap.add_argument("--candidates", action="store_true",
                    help="дополнительно выгружать ходы проваленных задач без сильного гейта "
                         "как кандидатов (label = -1) — пул для разбора судьёй")
    ap.add_argument("--max-chars", type=int, default=200_000,
                    help="пропускать примеры длиннее — они не влезают в бюджет судьи")
    args = ap.parse_args()

    csv.field_size_limit(2**31 - 1)
    user_tools = load_known_user_tools()
    files = [f for f in sorted(Path(args.data).glob("*.json"))
             if f.stat().st_size <= args.max_mb * 1e6][: args.limit_files]
    print(f"файлов: {len(files)}")

    stats: collections.Counter = collections.Counter()
    by_dom: collections.Counter = collections.Counter()
    convs: set[str] = set()
    n_rows = n_pos = n_cand = 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        fh,
        fieldnames=["id", "source_file", "sim_index", "turn_idx", "trial", "label",
                    "label_source", "gates", "domain", "reward", "has_tool_calls",
                    "prompt_chars", "response_chars"],
    )
    writer.writeheader()

    for f in files:
        # Файл читается ОДИН раз, а не по разу на каждое значение reward.
        # `[инцидент 28.08.2026]` Прежняя версия копила все строки в памяти и читала
        # каждый файл дважды; на banking-файлах по 250 МБ процесс раздулся до 14.5 ГБ
        # и был убит OOM-killer, не оставив ни строки результата. Теперь строки пишутся
        # потоком, а разбор идёт одним проходом.
        for reward, base_label, base_source in ((None, None, None),):
            try:
                for tt in iter_turns(f, only_reward=reward, max_sims=args.max_sims,
                                     require_tool_calls=not args.text_turns):
                    if tt.reward == 1.0:
                        base_label, base_source = 0, "solved_no_gate"
                    elif tt.reward == 0.0:
                        base_label, base_source = 1, "failed_gate_fired"
                    else:
                        stats["пропущено: промежуточный reward"] += 1
                        continue
                    # Метка вычисляется заново на КАЖДОМ ходу. Переиспользовать переменную
                    # цикла нельзя: одно присваивание внутри тела испортит все следующие
                    # итерации (проверено — девять позитивов молча превратились в кандидатов).
                    label, source = base_label, base_source
                    rep = apply_gates(
                        tt.transcript, tt.turn, user_tools=user_tools.get(tt.domain, set())
                    )
                    if rep.silenced:
                        stats["пропущено: нездоровый разбор"] += 1
                        continue
                    if label == 0 and rep.fired:
                        stats["пропущено: решённая, но гейт сработал"] += 1
                        continue
                    # Текстовый ход из проваленной задачи автоматически становится
                    # кандидатом: ни один гейт до него не достаёт, потому что все они
                    # требуют вызова инструмента. Позитивов из этой страты машинно
                    # не добыть — только судьёй по пулу кандидатов.
                    if label == 1 and not any(g in POSITIVE_GATES for g in rep.names):
                        # Ход проваленной задачи, который детерминированный слой не объяснил.
                        # Это не негатив и не позитив, а КАНДИДАТ: именно здесь живут
                        # категории, под которые у нас нет машинной проверки — ложный отказ,
                        # неверный расчёт по политике, выдуманный факт в тексте. Разбирать
                        # такой пул судьёй в разы дешевле, чем сканировать всё подряд.
                        stats["кандидат: проваленная без сильного гейта"] += 1
                        if not args.candidates:
                            continue
                        label, source = -1, "failed_unlabeled_candidate"

                    prompt, response = render_example(tt)
                    if len(prompt) > args.max_chars:
                        stats["пропущено: слишком длинный"] += 1
                        continue
                    # В индекс идут ССЫЛКИ и признаки, а не текст. Текст восстанавливается
                    # функцией `render_by_ref`. Первая версия писала полный промпт и дала
                    # файл на 9.9 ГБ, непригодный к работе.
                    writer.writerow(
                        {
                            "id": f"{tt.uid}#{tt.trial}",
                            "source_file": tt.source,
                            "sim_index": tt.sim_index,
                            "turn_idx": tt.turn_idx,
                            "trial": tt.trial,
                            "label": label,
                            "label_source": source,
                            "gates": "|".join(rep.names),
                            "domain": tt.domain,
                            "reward": tt.reward,
                            "has_tool_calls": int(bool(tt.turn.tool_calls)),
                            "prompt_chars": len(prompt),
                            "response_chars": len(response),
                        }
                    )
                    n_rows += 1
                    n_pos += label == 1
                    n_cand += label == -1
                    by_dom[tt.domain] += 1
                    convs.add(tt.uid.split("::")[0])
                    stats[f"взято: {source}"] += 1
            except Exception as e:  # noqa: BLE001 — один битый файл не должен ронять сборку
                print(f"  ПРОПУЩЕН {f.name[:50]}: {type(e).__name__}: {e}", flush=True)
        fh.flush()
        gc.collect()  # banking-файлы по 250 МБ: без явной сборки пик растёт от файла к файлу
        print(f"  {f.name[:62]:62s} строк всего: {n_rows}", flush=True)

    fh.close()
    out = Path(args.out)

    print(f"\nзаписано {n_rows} строк → {out}")
    neg = n_rows - n_pos - n_cand
    print(f"  позитивов {n_pos}, негативов {neg}, кандидатов {n_cand}")
    if n_pos + neg:
        print(f"  доля позитива среди размеченных: {n_pos/(n_pos+neg):.3f}")
    print("  по доменам:", dict(by_dom))
    print("  диалогов:", len(convs))
    print("\nучёт отбраковки:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    print(
        "\n⚠️ Метки слабые и выведены из reward и гейтов. Годятся для относительных сравнений\n"
        "   и для обучения, но НЕ как эталон абсолютного качества — см. docstring."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

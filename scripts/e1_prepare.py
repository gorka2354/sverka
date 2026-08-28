"""E1, шаг 1 — подготовить 46 выданных примеров для прогона стокового ревьюера τ²-bench.

Гипотеза E1: организаторы размечали данные ревьюером `review_llm_judge.py` из tau2-bench
или его вариантом. Если согласие с эталонными метками высокое, функция метки известна,
и собственный корпус размечается автоматически по публичным траекториям — тогда inc-2
меняется целиком.

**Режим прогона — чёрный ящик, и это осознанный отход от стокового.** Стоковый ревьюер
получает приватную истину задачи: `<User Instructions>`, `<Example Action Trajectory>`,
`<Natural Language Assertions>`. Ни одно из 23 объяснений выданного сэмпла на неё не
ссылается — все аргументируют только от политики и видимой истории. Поэтому эти поля
заполняются заглушкой: проверяется рубрика, а не доступ к ответам. Это же совпадает
с нашей боевой постановкой, где никакой приватной истины нет.

**Контроль интерпретируемости.** Прогон идёт на слабой локальной модели, и низкое согласие
само по себе не различает «рубрика другая» и «модель не тянет». Поэтому каждый пример
помечается флагом `gate_fired`: на этих строках ответ уже известен детерминированно
(вызван недоступный инструмент, повтор провалившегося вызова, заглушка в аргументе).
Если модель не находит ошибку даже там — упирается модель, и эксперимент читается
именно так, а не как опровержение гипотезы.

    python scripts/e1_prepare.py --out e1_input.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sverka.data import load_examples  # noqa: E402
from sverka.gates import apply_gates, load_known_user_tools  # noqa: E402
from sverka.transcript import USER, parse_response, parse_transcript  # noqa: E402

PLACEHOLDER = "(недоступно: прогон в режиме чёрного ящика, приватная истина задачи не выдаётся)"


def render_conversation(tr, reviewed_turn) -> tuple[str, int]:
    """Собрать текст разговора с явными номерами ходов.

    Стоковый ревьюер возвращает `turn_idx`, поэтому нумерация обязана быть видимой ему
    и совпадать с тем, по чему мы потом ищем вердикт. Проверяемый ход всегда последний.
    """
    lines: list[str] = []
    idx = 0
    for t in tr.turns:
        if t.role == "SYSTEM":
            continue
        who = "User" if t.role == USER else "Agent"
        body = (t.text or "").strip()
        if t.tool_calls:
            calls = "\n".join(
                f"    tool_call {c.name}({json.dumps(c.args, ensure_ascii=False)})"
                for c in t.tool_calls
            )
            body = (body + "\n" + calls).strip()
        for name, res in t.tool_responses:
            body += f"\n    tool_response {name}: {str(res)[:2000]}"
        lines.append(f"[turn {idx}] {who}: {body}".rstrip())
        idx += 1

    body = (reviewed_turn.text or "").strip()
    if reviewed_turn.tool_calls:
        calls = "\n".join(
            f"    tool_call {c.name}({json.dumps(c.args, ensure_ascii=False)})"
            for c in reviewed_turn.tool_calls
        )
        body = (body + "\n" + calls).strip()
    reviewed_idx = idx
    lines.append(f"[turn {reviewed_idx}] Agent: {body}".rstrip())
    return "\n".join(lines), reviewed_idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/valid.parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    user_tools = load_known_user_tools()
    rows = []
    for e in load_examples(args.data):
        tr = parse_transcript(e.prompt, strict=False)
        turn = parse_response(e.response, strict=False)
        rep = apply_gates(tr, turn, user_tools=user_tools.get(e.domain, set()))
        conversation, reviewed_idx = render_conversation(tr, turn)
        rows.append(
            {
                "id": e.id,
                "domain": e.domain,
                "label": e.label,
                "policy": tr.policy or tr.system,
                "conversation": conversation,
                "reviewed_turn_idx": reviewed_idx,
                "gate_fired": rep.fired,
                "gates": rep.names,
                "n_chars": len(conversation),
            }
        )

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    fired = sum(1 for r in rows if r["gate_fired"])
    chars = sorted(r["n_chars"] for r in rows)
    print(f"записано {len(rows)} примеров в {out}")
    print(f"  из них с сработавшим гейтом (контроль интерпретируемости): {fired}")
    print(f"  длина разговора: p50={chars[len(chars)//2]} p90={chars[int(.9*len(chars))]} max={chars[-1]}")
    print(f"  оценка в токенах (3.3 симв/ток): p50={chars[len(chars)//2]//3} max={chars[-1]//3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

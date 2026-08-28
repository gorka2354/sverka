"""Прогон судьи по материализованной выборке корпуса.

Судья — стоковый ревьюер tau2-bench, его системный промпт берётся из репозитория дословно.
Правки промпта проверены и отвергнуты: `[замер E1b]` явное указание проверяемого хода
с оговоркой «иначе верни пустой список» обрушило детекцию с F1 0.786 до 0.333 — для 9B это
оказалось не уточнением области, а разрешением молчать.

## Что именно меряется

Вердикт берётся в формулировке **«нашлась хоть какая-то ошибка агента»**, а не «ошибка
именно на проверяемом ходу». Причина замерена: на выданном сэмпле локализация даёт F1 0.435,
детекция — 0.765, и 8 из 12 пропусков это находки, приписанные другому ходу. Локализация
нашей задаче не нужна: контекст обрывается ровно перед проверяемым ходом.

## Зачем прогон по корпусу, если есть прогон по сэмплу

На выданных 46 примерах нельзя измерить долю ложных срабатываний: в страте «ответ
с вызовом» всего два достижимых отрицательных примера. В корпусе надёжных негативов
176 622, и ложные срабатывания наконец становятся измеримыми. Это главная цифра прогона.

По кандидатам меряется выход — какую долю пула судья помечает. Вместе с долей ложных
это даёт оценку числа настоящих позитивов среди помеченных.

    python scripts/judge_run.py --sample sample.csv --out judged.jsonl --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PLACEHOLDER = "(not available: black-box run, private task ground truth is withheld)"
REPO_JUDGE = "~/sverka/tau2-bench/src/tau2/evaluator/review_llm_judge.py"


def load_stock_prompts(path: str) -> tuple[str, str]:
    """Системный и пользовательский промпт стокового ревьюера — дословно из репозитория.

    Читается из исходника, а не копируется в наш код: проверяется ИХ рубрика, и пересказ
    здесь означал бы, что мы меряем свой пересказ.
    """
    src = Path(path).expanduser().read_text(encoding="utf-8")
    sys_p = src.split('SYSTEM_PROMPT = """', 1)[1].split('""".strip()', 1)[0].strip()
    usr_p = src.split('USER_PROMPT = """', 1)[1].split('""".strip()', 1)[0].strip()
    return sys_p, usr_p


def extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = m.group(1) if m else None
    if blob is None:
        i, j = text.find("{"), text.rfind("}")
        blob = text[i: j + 1] if i >= 0 and j > i else None
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:  # noqa: BLE001
        return None


def build_conversation(prompt: str, response: str) -> tuple[str, str, int]:
    """Из нашего формата собрать (политика, разговор, номер проверяемого хода)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sverka.transcript import USER, parse_response, parse_transcript

    tr = parse_transcript(prompt, strict=False)
    turn = parse_response(response, strict=False)
    lines: list[str] = []
    idx = 0
    for t in tr.turns:
        if t.role == "SYSTEM":
            continue
        who = "User" if t.role == USER else "Agent"
        body = (t.text or "").strip()
        for c in t.tool_calls:
            body += f"\n    tool_call {c.name}({json.dumps(c.args, ensure_ascii=False)})"
        for name, res in t.tool_responses:
            body += f"\n    tool_response {name}: {str(res)[:2000]}"
        lines.append(f"[turn {idx}] {who}: {body}".rstrip())
        idx += 1
    body = (turn.text or "").strip()
    for c in turn.tool_calls:
        body += f"\n    tool_call {c.name}({json.dumps(c.args, ensure_ascii=False)})"
    lines.append(f"[turn {idx}] Agent: {body}".rstrip())
    return (tr.policy or tr.system), "\n".join(lines), idx


def ask(model: str, system: str, user: str, n_ctx: int, timeout: int = 3600):
    body = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        # think=False обязателен: у thinking-моделей весь бюджет генерации уходит
        # в рассуждение, а поле response приходит ПУСТЫМ. `[замер]` без флага валидный
        # JSON получался у 4 запросов из 14; с ним время на пример упало со 100 с до 38.
        "think": False,
        "options": {"num_ctx": n_ctx, "num_predict": 1200, "temperature": 0},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--judge-src", default=REPO_JUDGE)
    ap.add_argument("--max-ctx", type=int, default=32768)
    args = ap.parse_args()

    csv.field_size_limit(2**31 - 1)
    system_prompt, user_tpl = load_stock_prompts(args.judge_src)
    print(f"системный промпт: {len(system_prompt)} символов; модель {args.model}")

    with open(args.sample, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: len(r["prompt"]))

    done: set[str] = set()
    if os.path.exists(args.out):
        for line in Path(args.out).read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    print(f"примеров: {len(rows)}, уже посчитано: {len(done)}")

    with open(args.out, "a", encoding="utf-8") as out:
        for k, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            policy, conversation, reviewed = build_conversation(r["prompt"], r["response"])
            user = user_tpl.format(
                policy=policy,
                simulation_guidelines=PLACEHOLDER,
                user_instructions=PLACEHOLDER,
                example_action_trajectory=PLACEHOLDER,
                natural_language_assertions=PLACEHOLDER,
                conversation=conversation,
            )
            est = (len(system_prompt) + len(user)) // 3 + 1200
            n_ctx = min(args.max_ctx, max(8192, ((est // 4096) + 1) * 4096))
            rec = {"id": r["id"], "label": r["label"], "label_source": r["label_source"],
                   "domain": r["domain"], "gates": r["gates"],
                   "has_tool_calls": r["has_tool_calls"], "turn_idx": r["turn_idx"],
                   "est_tokens": est, "n_ctx": n_ctx, "truncated_ctx": est > n_ctx}
            t0 = time.time()
            try:
                resp = ask(args.model, system_prompt, user, n_ctx)
                txt = resp.get("response", "") or ""
                parsed = extract_json(txt)
                rec["json_ok"] = parsed is not None
                if parsed:
                    errs = [e for e in (parsed.get("errors") or []) if isinstance(e, dict)]
                    agent = [e for e in errs if str(e.get("source", "")).lower() == "agent"]
                    at_turn = [e for e in agent if e.get("turn_idx") == reviewed]
                    rec["pred_any_agent_error"] = int(bool(agent))
                    rec["pred_at_reviewed_turn"] = int(bool(at_turn))
                    rec["tags"] = sorted({t for e in agent for t in (e.get("error_tags") or [])})
                else:
                    rec["pred_any_agent_error"] = None
                    rec["pred_at_reviewed_turn"] = None
                    rec["raw_head"] = txt[:300]
            except Exception as e:  # noqa: BLE001 — один пример не должен ронять прогон
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{k}/{len(rows)}] {r['id'][:40]:40s} метка={r['label']:>2s} "
                  f"ctx={n_ctx} {rec['seconds']}c вердикт={rec.get('pred_any_agent_error')}",
                  flush=True)
    print("=== ГОТОВО ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

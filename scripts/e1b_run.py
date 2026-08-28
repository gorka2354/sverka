# -*- coding: utf-8 -*-
"""E1, шаг 2 — прогон стокового ревьюера tau2-bench через ollama на мини-ПК.

Системный промпт берётся ДОСЛОВНО из репозитория tau2-bench: проверяется именно их рубрика,
а не наш пересказ. Пользовательский промпт заполняется политикой и разговором; поля с
приватной истиной задачи (User Instructions / Example Action Trajectory / NL Assertions)
заполняются заглушкой — прогон идёт в режиме чёрного ящика, как и наша боевая постановка.

Примеры сортируются по длине по возрастанию: если длинные не влезут в контекст, короткие
уже будут посчитаны, и эксперимент даст результат, а не ноль.
"""
import json, re, sys, time, urllib.request, os

REPO = os.path.expanduser("~/sverka/tau2-bench/src/tau2/evaluator/review_llm_judge.py")
IN = os.path.expanduser("~/sverka/e1/e1_input.jsonl")
OUT = os.path.expanduser("~/sverka/e1/e1b_results.jsonl")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:9b"
MAX_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
PLACEHOLDER = "(not available: black-box run, private task ground truth is withheld)"

src = open(REPO, encoding="utf-8").read()
SYSTEM_PROMPT = src.split('SYSTEM_PROMPT = """', 1)[1].split('""".strip()', 1)[0].strip()
USER_PROMPT = src.split('USER_PROMPT = """', 1)[1].split('""".strip()', 1)[0].strip()
print(f"системный промпт: {len(SYSTEM_PROMPT)} символов; модель {MODEL}; ctx {MAX_CTX}", flush=True)


def extract_json(text):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = m.group(1) if m else None
    if blob is None:
        i, j = text.find("{"), text.rfind("}")
        blob = text[i : j + 1] if i >= 0 and j > i else None
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


def ask(system, user, n_ctx):
    body = {
        "model": MODEL,
        "system": system,
        "prompt": user,
        "stream": False,
        # think=False обязателен: qwen3.5 — thinking-модель, и с включённым рассуждением
        # весь бюджет генерации уходит туда, а поле response приходит ПУСТЫМ.
        # Замер: без него валидный JSON получался у 4 запросов из 14.
        "think": False,
        "options": {"num_ctx": n_ctx, "num_predict": 1200, "temperature": 0},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: r["n_chars"])
done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        try:
            done.add(json.loads(l)["id"])
        except Exception:
            pass
print(f"примеров: {len(rows)}, уже посчитано: {len(done)}", flush=True)

out = open(OUT, "a", encoding="utf-8")
for k, r in enumerate(rows, 1):
    if r["id"] in done:
        continue
    user = USER_PROMPT.format(
        policy=r["policy"],
        simulation_guidelines=PLACEHOLDER,
        user_instructions=PLACEHOLDER,
        example_action_trajectory=PLACEHOLDER,
        natural_language_assertions=PLACEHOLDER,
        conversation=r["conversation"],
    )
    user += (
        "\n\n<Turn Under Review>\n"
        f"Only turn {r['reviewed_turn_idx']} is under review. It is the FINAL Agent turn in "
        "the conversation above. Earlier turns are context only: an error the agent made "
        "in an EARLIER turn is NOT an error of the turn under review, even if its "
        "consequences persist.\n"
        f"If and only if the agent made a NEW error in turn {r['reviewed_turn_idx']}, "
        f"report it with turn_idx = {r['reviewed_turn_idx']}. Otherwise return an empty errors list.\n"
        "</Turn Under Review>"
    )
    est = (len(SYSTEM_PROMPT) + len(user)) // 3 + 1200
    n_ctx = min(MAX_CTX, max(8192, ((est // 4096) + 1) * 4096))
    t0 = time.time()
    rec = {"id": r["id"], "label": r["label"], "gate_fired": r["gate_fired"],
           "domain": r["domain"], "est_tokens": est, "n_ctx": n_ctx}
    try:
        resp = ask(SYSTEM_PROMPT, user, n_ctx)
        txt = resp.get("response", "") or ""
        parsed = extract_json(txt)
        rec["raw_len"] = len(txt)
        rec["json_ok"] = parsed is not None
        rec["truncated_ctx"] = est > n_ctx
        if parsed:
            errs = [e for e in (parsed.get("errors") or []) if isinstance(e, dict)]
            agent_errs = [e for e in errs if str(e.get("source", "")).lower() == "agent"]
            at_turn = [e for e in agent_errs if e.get("turn_idx") == r["reviewed_turn_idx"]]
            rec["pred_any_agent_error"] = int(bool(agent_errs))
            rec["pred_at_reviewed_turn"] = int(bool(at_turn))
            rec["tags"] = sorted({t for e in at_turn for t in (e.get("error_tags") or [])})
            rec["n_errors"] = len(errs)
        else:
            rec["pred_any_agent_error"] = None
            rec["pred_at_reviewed_turn"] = None
            rec["raw_head"] = txt[:400]
        rec["prompt_eval_count"] = resp.get("prompt_eval_count")
        rec["seconds"] = round(time.time() - t0, 1)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["seconds"] = round(time.time() - t0, 1)
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()
    print(f"[{k}/{len(rows)}] {r['id'][:44]:44s} ctx={n_ctx} {rec.get('seconds')}c "
          f"json={rec.get('json_ok')} pred={rec.get('pred_at_reviewed_turn')} gold={r['label']}",
          flush=True)
out.close()
print("=== E1 ГОТОВО ===", flush=True)

"""Benchmark evaluation on a running LlamaServer: generate -> extract -> execute -> per-task outcomes.

Also classifies every failure into a taxonomy so we can see *why* the model fails (the obstacle-removal question).
"""
import re
import time

from . import bench
from .engine import strip_bos
from .execute import run_programs

MAX_TOKENS = {"humaneval+": 8192, "mbpp+": 8192, "humanevalfix": 8192}  # no practical truncation (user requirement)


THOUGHT = re.compile(r"<\|channel>.*?(<channel\|>|$)", re.S)


def strip_thought(text):
    """Removes Gemma 4's thought channel so code extraction only sees the final answer."""
    return THOUGHT.sub("", text)


def render(tok, users, system=None, thinking=False):
    out = []
    kw = dict(enable_thinking=True) if thinking else {}
    for u in users:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
        out.append(strip_bos(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)))
    return out


def classify(task, gen, code, exec_res):
    """Failure taxonomy (mutually exclusive, checked in order)."""
    if exec_res["passed"]:
        return "pass"
    text = gen["text"]
    if gen.get("stop_type") == "limit":
        # degenerate repetition vs. long output
        tail = text[-600:]
        if len(tail) > 200 and len(set(tail.split())) < 0.2 * max(1, len(tail.split())):
            return "loop"
        return "truncated"
    if "```" not in text:
        return "no_code_block"
    ep = task.get("entry_point")
    if ep and not re.search(rf"def\s+{re.escape(ep)}\s*\(", code) and task["bench"] != "humaneval+":
        return "wrong_or_missing_entry_point"
    err = exec_res.get("stderr_tail", "")
    if exec_res["status"] == "timeout":
        return "timeout"
    if "SyntaxError" in err or "IndentationError" in err:
        return "syntax_error"
    if "NameError" in err or "ImportError" in err or "ModuleNotFoundError" in err:
        return "name_or_import_error"
    if "AssertionError" in err:
        return "wrong_answer"
    if "Error" in err or "Exception" in err:
        return "runtime_error"
    return "other_fail"


def run_suite(server, tok, tasks, system=None, lora_scales=None, max_tokens=None, sampling=None, exec_workers=8,
              thinking=False):
    prompts = render(tok, [t["user"] for t in tasks], system, thinking)
    mt = max_tokens or MAX_TOKENS[tasks[0]["bench"]]
    t0 = time.time()
    gens, gstats = server.complete_many(prompts, max_tokens=mt, lora_scales=lora_scales, sampling=sampling)
    for g in gens:
        g["raw"] = g["text"]
        g["text"] = strip_thought(g["text"])
    codes = [bench.extract_code(g["text"], t.get("entry_point")) for t, g in zip(tasks, gens)]
    progs = [(t["task_id"], bench.build_program(t, c)) for t, c in zip(tasks, codes)]
    t1 = time.time()
    ex = run_programs(progs, workers=exec_workers)
    rows = []
    for t, g, c in zip(tasks, gens, codes):
        e = ex[t["task_id"]]
        rows.append(dict(task_id=t["task_id"], passed=e["passed"], status=e["status"],
                         category=classify(t, g, c, e), n_gen=g["n_gen"], n_prompt=g["n_prompt"],
                         stop_type=g["stop_type"], response=g["raw"], stderr_tail=e["stderr_tail"][-600:]))
    n = len(rows)
    k = sum(r["passed"] for r in rows)
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    summary = dict(bench=tasks[0]["bench"], n=n, passed=k, pass_rate=round(k / n, 4), categories=cats,
                   mean_gen_tokens=round(sum(r["n_gen"] or 0 for r in rows) / n, 1),
                   gen=gstats, exec_s=round(time.time() - t1, 1), total_s=round(time.time() - t0, 1))
    return summary, rows


def mmlu_eval(server, tok, items, system=None, lora_scales=None):
    """Letter-answer accuracy with greedy decoding of up to 4 tokens (robust to formatting)."""
    prompts = render(tok, [it["user"] for it in items], system)
    gens, gstats = server.complete_many(prompts, max_tokens=64, lora_scales=lora_scales)
    rows = []
    for it, g in zip(items, gens):
        m = re.search(r"\b([ABCD])\b", strip_thought(g["text"]))
        pred = "ABCD".index(m.group(1)) if m else -1
        rows.append(dict(task_id=it["task_id"], passed=pred == it["answer"], pred=pred, response=g["text"]))
    k = sum(r["passed"] for r in rows)
    return dict(bench="mmlu", n=len(rows), passed=k, pass_rate=round(k / len(rows), 4),
                unparsed=sum(r["pred"] == -1 for r in rows), gen=gstats), rows


def run_repopatch(server, tok, tasks, system=None, lora_scales=None, max_tokens=8192, sampling=None):
    from . import repobench
    prompts = render(tok, [repobench.render_task(t) for t in tasks], system)
    t0 = time.time()
    gens, gstats = server.complete_many(prompts, max_tokens=max_tokens, lora_scales=lora_scales, sampling=sampling)
    scored = [repobench.score(t, g["text"]) for t, g in zip(tasks, gens)]
    progs = [(t["task_id"], s["program"]) for t, s in zip(tasks, scored) if s["program"]]
    ex = run_programs(progs)
    rows = []
    for t, g, s in zip(tasks, gens, scored):
        e = ex.get(t["task_id"], {"passed": False, "status": "not_run", "stderr_tail": ""})
        cat = "pass" if e["passed"] else ("format_error" if not s["format_ok"] else
                                          ("edit_not_applicable" if not s["applied"] else
                                           ("truncated" if g.get("stop_type") == "limit" else "tests_fail")))
        rows.append(dict(task_id=t["task_id"], passed=e["passed"], format_ok=s["format_ok"], applied=s["applied"],
                         category=cat, n_gen=g["n_gen"], stop_type=g["stop_type"], response=g["text"],
                         stderr_tail=e["stderr_tail"][-400:]))
    n = len(rows)
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    summ = dict(bench="repopatch", n=n, passed=sum(r["passed"] for r in rows),
                pass_rate=round(sum(r["passed"] for r in rows) / n, 4),
                format_ok_rate=round(sum(r["format_ok"] for r in rows) / n, 4),
                applied_rate=round(sum(r["applied"] for r in rows) / n, 4), categories=cats,
                mean_gen_tokens=round(sum(r["n_gen"] or 0 for r in rows) / n, 1), gen=gstats,
                total_s=round(time.time() - t0, 1))
    return summ, rows

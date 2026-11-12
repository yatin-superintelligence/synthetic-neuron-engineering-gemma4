"""Minimal agentic repair loop (execution feedback), mirroring how SWE agents iterate against tests.

Turn 1: the task prompt. After each answer the code is run against the VISIBLE tests only. If they fail, the
error output is returned to the model as a new user turn and it may revise (up to max_turns). The final answer is
scored on the HIDDEN tests. Supported suites: humanevalfix (visible = docstring examples), mbpp_* (visible = the
single assert shown in the prompt).
"""
from . import bench
from .engine import strip_bos
from .execute import run_programs

FEEDBACK = ("I ran the example tests on your function and they failed:\n```\n{err}\n```\n"
            "Fix the function. Respond with the complete corrected function in a single ```python code block.")


def _render(tok, msgs, system=None):
    m = ([{"role": "system", "content": system}] if system else []) + msgs
    return strip_bos(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))


def repair_loop(server, tok, tasks, max_turns=3, system=None, lora_scales=None, max_tokens=8192):
    convs = {t["task_id"]: [{"role": "user", "content": t["user"]}] for t in tasks}
    final_code, turns_used, visible_ok = {}, {}, {}
    active = list(tasks)
    stats = []
    for turn in range(1, max_turns + 1):
        prompts = [_render(tok, convs[t["task_id"]], system) for t in active]
        gens, st = server.complete_many(prompts, max_tokens=max_tokens, lora_scales=lora_scales)
        stats.append(st)
        codes = {t["task_id"]: bench.extract_code(g["text"], t.get("entry_point")) for t, g in zip(active, gens)}
        vis = run_programs([(t["task_id"], bench.build_program(t, codes[t["task_id"]], visible=True))
                            for t in active])
        nxt = []
        for t, g in zip(active, gens):
            tid = t["task_id"]
            convs[tid].append({"role": "assistant", "content": g["text"]})
            final_code[tid], turns_used[tid] = codes[tid], turn
            visible_ok[tid] = vis[tid]["passed"]
            if not vis[tid]["passed"] and turn < max_turns:
                err = vis[tid]["stderr_tail"].strip()[-800:] or vis[tid]["status"]
                convs[tid].append({"role": "user", "content": FEEDBACK.format(err=err)})
                nxt.append(t)
        active = nxt
        if not active:
            break
    hidden = run_programs([(t["task_id"], bench.build_program(t, final_code[t["task_id"]])) for t in tasks])
    rows = []
    for t in tasks:
        tid = t["task_id"]
        rows.append(dict(task_id=tid, passed=hidden[tid]["passed"], visible_passed=visible_ok[tid],
                         turns=turns_used[tid], conversation=convs[tid],
                         hidden_stderr=hidden[tid]["stderr_tail"][-400:]))
    n = len(rows)
    first_turn_pass = None
    summ = dict(n=n, passed=sum(r["passed"] for r in rows), pass_rate=round(sum(r["passed"] for r in rows) / n, 4),
                visible_pass_rate=round(sum(r["visible_passed"] for r in rows) / n, 4),
                false_confidence=sum(r["visible_passed"] and not r["passed"] for r in rows),
                mean_turns=round(sum(r["turns"] for r in rows) / n, 3),
                gen_tokens=sum(s["gen_tokens"] for s in stats), wall_s=round(sum(s["wall_s"] for s in stats), 1))
    return summ, rows

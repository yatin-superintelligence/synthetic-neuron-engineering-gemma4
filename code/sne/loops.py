"""Thinking-loop detection and LoopBench: replay the agent state just before a response, score whether it loops.

Source traces are official harness runs (ATIF). They contain competition task text, so derived contexts are stored only
under the git-ignored data/kaggle/ folder; committed outputs hold task ids and numeric scores.

A response is a LOOP when it repeats one whole sentence at least LOOP_MIN_REPEATS times and its word 3-gram repetition is
at least LOOP_MIN_REP3. Both thresholds were fixed from the first census before any edit was evaluated.
"""
import collections
import glob
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from . import DATA, SMOKE

LOOP_MIN_REPEATS = 8
LOOP_MIN_REP3 = 0.5
LOOPBENCH = os.path.join(DATA, "kaggle", "loopbench_v1.jsonl")
THINK_BUDGET, MAX_OUTPUT, SLOT_CTX = 4096, 16384, 32768  # sample-submission defaults and vLLM max_model_len
STOP = ["<|tool_response>"]  # vLLM stops on generation_config eos ids, which include this token; llama.cpp does not
SLOTS = 1  # replay slots: 1 slot took 47 s per state, 2 to 6 slots took 84 to 89 s (EXP-057)
SERVER_ARGS = ["--special"]  # keeps <channel|> and <|tool_call> in the text, which the scoring below reads


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?:\n])\s+", text) if len(s.strip()) > 25]


def rep3(text):
    w = text.split()
    if len(w) < 40:
        return 0.0
    tri = list(zip(w, w[1:], w[2:]))
    return 1 - len(set(tri)) / len(tri)


def loop_stats(text):
    ss = sentences(text)
    cnt = collections.Counter(ss)
    top, n = cnt.most_common(1)[0] if cnt else ("", 0)
    onset = None
    if n >= 3:
        seen = set()
        pos = 0
        for s in ss:
            if cnt[s] >= 3 and s in seen:
                onset = pos
                break
            seen.add(s)
            pos += len(s) + 1
    r = rep3(text)
    return dict(chars=len(text), rep3=round(r, 4), max_sentence_repeats=n, onset_char=onset,
                loop=n >= LOOP_MIN_REPEATS and r >= LOOP_MIN_REP3)


def agent_steps(trace):
    return [(i, s) for i, s in enumerate(trace.get("steps") or []) if s.get("source") == "agent"]


def census(trace_dir):
    """One row per task: loops per response, tokens, what preceded the first loop."""
    rows = []
    for f in sorted(glob.glob(os.path.join(trace_dir, "trace_*.json"))):
        t = json.load(open(f, encoding="utf-8"))
        iid = os.path.basename(f)[6:-5]
        loops, n, gen = [], 0, 0
        for i, s in agent_steps(t):
            n += 1
            gen += (s.get("metrics") or {}).get("completion_tokens") or 0
            st = loop_stats(s.get("message") or "")
            if st["loop"]:
                prev = t["steps"][i - 1]
                obs = json.dumps(prev.get("observation") or "")[:400] if prev.get("source") == "agent" else ""
                loops.append(dict(step=i, tokens=(s.get("metrics") or {}).get("completion_tokens"), prev_tool_error='"status": "error"' in obs.replace('\\"', '"'),
                                  **{k: st[k] for k in ("rep3", "max_sentence_repeats", "onset_char", "chars")}))
        rows.append(dict(instance_id=iid, responses=n, completion_tokens=gen, n_loops=len(loops), loops=loops))
    return rows


TOOL_DECLS = [
    ("run_command", "Executes a shell command in /bin/bash -c inside /workspace.", {"command": "string"}, ["command"]),
    ("submit_patch", "Captures the current diff as the final patch.", {}, []),
    ("get_status", "Returns live budget consumption and patch status.", {}, []),
    ("read_file", "Reads a file from /workspace with 1-indexed inclusive line slicing.",
     {"filepath": "string", "start_line": "integer", "end_line": "integer"}, ["filepath"]),
    ("edit_file", "Replaces old_string with new_string in an existing file.",
     {"filepath": "string", "old_string": "string", "new_string": "string", "allow_multiple": "boolean"},
     ["filepath", "old_string", "new_string"]),
    ("write_file", "Creates or overwrites a file.", {"filepath": "string", "content": "string"}, ["filepath", "content"]),
    ("get_code_neighbors", "Finds neighbors of a symbol in the call graph.",
     {"node": "string", "edge_type": "string", "max_neighbors": "integer"}, ["node"]),
    ("search_similar_code", "Finds graph nodes similar to query.", {"query": "string", "k": "integer"}, ["query"]),
    ("get_code_subgraph", "Extracts the induced subgraph for symbols.", {"nodes": "string"}, ["nodes"]),
]
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {k: {"type": v, "description": k} for k, v in p.items()}, "required": r}}}
    for n, d, p, r in TOOL_DECLS]


def messages_before(trace, idx):
    """Conversation up to, not including, step idx, in chat-template form, with prior thoughts kept as reasoning."""
    msgs = []
    for s in trace["steps"][:idx]:
        src = s.get("source")
        if src in ("system", "user"):
            msgs.append({"role": src, "content": s.get("message") or ""})
        elif src == "agent":
            calls = [{"type": "function", "function": {"name": c.get("function_name"), "arguments": c.get("arguments") or {}}}
                     for c in (s.get("tool_calls") or [])]
            msgs.append({"role": "assistant", "content": "" if calls else (s.get("message") or ""),
                         **({"reasoning": s.get("message") or ""} if calls else {}), **({"tool_calls": calls} if calls else {})})
            obs = s.get("observation")
            if calls and obs:
                text = obs.get("content") if isinstance(obs, dict) else json.dumps(obs)
                msgs.append({"role": "tool", "name": calls[0]["function"]["name"], "content": str(text)})
    return msgs


def split_map(seed=20261106):
    """One dev and test assignment for all 58 tasks, shared by every replay benchmark, so no task leaks across."""
    saved = os.path.join(DATA, "kaggle", "split_map.json")
    if os.path.exists(saved):  # the saved copy is the split of record: a snapshots folder holding only some tasks (after the
        return json.load(open(saved, encoding="utf-8"))  # 2026-11-10 storage cleanup) would otherwise give a different split
    snap = os.path.join(DATA, "kaggle", "snapshots")
    ids = sorted(f[:-4] for f in os.listdir(snap) if f.endswith(".tgz"))
    random.Random(seed).shuffle(ids)
    return {t: "dev" if i < len(ids) // 2 else "test" for i, t in enumerate(ids)}


def build(trace_dirs, n_controls=30, seed=20261106):
    """Loop contexts plus the same number of control contexts (non-loop responses from other steps)."""
    rng = random.Random(seed)
    loops, ctrls = [], []
    for run, d in trace_dirs.items():
        for f in sorted(glob.glob(os.path.join(d, "trace_*.json"))):
            t = json.load(open(f, encoding="utf-8"))
            iid = os.path.basename(f)[6:-5]
            for i, s in agent_steps(t):
                st = loop_stats(s.get("message") or "")
                if i < 2:
                    continue
                rec = dict(case=f"{run}/{iid}/{i}", run=run, instance_id=iid, step=i, kind="loop" if st["loop"] else "control",
                           messages=messages_before(t, i), original_rep3=st["rep3"], original_repeats=st["max_sentence_repeats"],
                           original_chars=st["chars"], onset_char=st["onset_char"], original_text=(s.get("message") or "")[:24000],
                           completion_tokens=(s.get("metrics") or {}).get("completion_tokens"))
                (loops if st["loop"] else ctrls).append(rec)
    rng.shuffle(ctrls)
    split = split_map()
    out = loops + ctrls[:n_controls]
    for r in out:
        r["split"] = split[r["instance_id"]]
    with open(LOOPBENCH, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def load(split=None, kind=None):
    rows = [json.loads(l) for l in open(LOOPBENCH, encoding="utf-8")]
    rows = [r for r in rows if (split is None or r["split"] == split) and (kind is None or r["kind"] == kind)]
    return rows[:6] if SMOKE else rows


def render(tok, case):
    s = tok.apply_chat_template(case["messages"], tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True, preserve_thinking=True)
    return s[len("<bos>"):] if s.startswith("<bos>") else s


def generate(server, prompt, n_prompt, sampling, lora_scales=None, budget=THINK_BUDGET, thinking=True, early_exit=False, chunk=1024):
    """One response along the official path: thinking is forced closed at `budget` tokens, as vLLM's
    ThinkingTokenBudgetLogitsProcessor does, and output continues up to the output limit. With early_exit a response
    stops once it already counts as a loop, because its label can no longer change; final evaluations leave this off."""
    text, n, closed, forced, stop = "", 0, not thinking, False, None
    limit = min(256 if SMOKE else MAX_OUTPUT, SLOT_CTX - n_prompt - 8)
    while n < limit:
        want = min(chunk, limit - n)
        if not closed:
            want = min(want, budget - n)
        g = server.complete(prompt + text, want, lora_scales, sampling)
        text += g["text"]
        n += g["n_gen"] or 0
        stop = g["stop_type"]
        closed = closed or "<channel|>" in text
        if stop != "limit" or not g["n_gen"]:
            break
        if early_exit and loop_stats(text)["loop"]:
            stop = "early_exit"
            break
        if not closed and n >= budget:
            text += "<channel|>"
            closed = forced = True
    return dict(text=text, n_gen=n, stop=stop, forced_close=forced, thought_closed="<channel|>" in text)


def run(server, tok, cases, lora_scales=None, seeds=(0,), temperature=0.2, top_p=0.95, extra=None, prompt_fn=None,
        budget=THINK_BUDGET, thinking=True, early_exit=False):
    """Replays loop contexts once per seed and controls once. Returns a summary, rows without text, and the texts."""
    if SMOKE:
        seeds, budget = seeds[:1], min(budget, 128)
    jobs, skipped = [], []
    for c in cases:
        p = (prompt_fn or render)(tok, c)
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        if n + 1024 > SLOT_CTX:  # no room to answer inside the official context window; reported, never silently dropped
            skipped.append(c["case"])
            continue
        jobs.append([(c, sd, p, n) for sd in (seeds if c["kind"] == "loop" else seeds[:1])])

    def one(group):  # one seed after another, so the slot reuses its prompt cache
        rows = []
        for c, sd, p, n in group:
            sampling = dict(temperature=temperature, top_p=top_p, top_k=0, seed=sd, stop=STOP, cache_prompt=True, **(extra or {}))
            g = generate(server, p, n, sampling, lora_scales, budget, thinking, early_exit)
            text = g.pop("text")
            rows.append((dict(case=c["case"], kind=c["kind"], split=c["split"], seed=sd, n_prompt=n, **g, tool_call="<|tool_call>" in text,
                              tool_call_closed="<tool_call|>" in text, **loop_stats(text)), text))
        return rows
    with ThreadPoolExecutor(server.parallel) as ex:
        out = [x for rows in ex.map(one, jobs) for x in rows]
    res = [r for r, _ in out]
    texts = {f"{r['case']}#{r['seed']}": t for r, t in out}

    def rate(kind, key):
        v = [r for r in res if r["kind"] == kind]
        return round(sum(bool(r[key]) for r in v) / max(1, len(v)), 4), len(v)
    gen = [r["n_gen"] or 0 for r in res]
    summ = dict(n=len(res), skipped_too_long=skipped, loop_rate_on_loop_contexts=rate("loop", "loop"),
                loop_rate_on_controls=rate("control", "loop"), tool_call_rate_on_loop_contexts=rate("loop", "tool_call"),
                tool_call_rate_on_controls=rate("control", "tool_call"), forced_close_on_loop_contexts=rate("loop", "forced_close"),
                forced_close_on_controls=rate("control", "forced_close"), mean_tokens=round(sum(gen) / max(1, len(gen)), 1))
    return summ, res, texts

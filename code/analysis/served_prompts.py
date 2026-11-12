"""Reads vLLM's request log of an official run (results/vllm_requests.log.gz, written when a run's config has log_requests) and pairs
every request with the agent step it produced, so edits can be fitted on exactly what the model was served.

vLLM 0.19.1 (vllm/entrypoints/logger.py) writes, per request id:
  Request <id> details: prompt: '<text>', prompt_token_ids: [..], prompt_embeds shape: None.      (the exact served prompt)
  Generated response <id>[ (streaming complete)]: output: '<text>', output_token_ids: [..]|None, finish_reason: <r>
The harness trace of each task holds, per agent step, metrics.prompt_tokens and metrics.completion_tokens. A request is matched to a
step when its prompt and output token counts equal the step's two recorded counts (ties are broken by the decoded output starting
the step's thought); the counts printed below report how many steps found exactly one request. After a compaction the prompt no
longer holds the task text, so the text is not used for matching.

Usage: python tools/analysis/served_prompts.py <run folder, e.g. artifacts_local/colab/exp-129a> [--write]
  prints the counts; --write saves served.jsonl.gz in the run folder: one row per matched step with task, step index, prompt ids,
  output ids when logged, output text.
"""
import ast
import collections
import glob
import gzip
import json
import os
import re
import sys

REQ = re.compile(r"^\S+ \S+ Request (\S+) details: prompt: (.*), prompt_token_ids: (\[[\d, ]*\]|None), prompt_embeds shape: ")
OUT = re.compile(r"^\S+ \S+ Generated response (\S+?)( \(streaming complete\)| \(streaming delta\))?: output: (.*), output_token_ids: (\[[\d, ]*\]|None), finish_reason: (\S+)$")


def read_log(path):
    """{request id: dict(prompt_ids, prompt_text, output_text, output_ids, finish)} in log order."""
    reqs = collections.OrderedDict()
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = REQ.match(line)
            if m:
                ids = ast.literal_eval(m.group(3))
                reqs[m.group(1)] = dict(prompt_ids=ids, prompt_text=ast.literal_eval(m.group(2)) if m.group(2) != "None" else None,
                                        output_text=None, output_ids=None, finish=None)
                continue
            m = OUT.match(line)
            if m and m.group(2) != " (streaming delta)":
                r = reqs.setdefault(m.group(1), dict(prompt_ids=None, prompt_text=None))
                r.update(output_text=ast.literal_eval(m.group(3)), output_ids=ast.literal_eval(m.group(4)), finish=m.group(5),
                         streamed=bool(m.group(2)))
    return reqs


def steps_of(trace):
    out = []
    for i, s in enumerate(trace.get("steps") or []):
        if s.get("source") != "agent":
            continue
        m = s.get("metrics") or {}
        if isinstance(m, str):
            m = ast.literal_eval(m)
        out.append(dict(step=i, prompt_tokens=m.get("prompt_tokens"), completion_tokens=m.get("completion_tokens"), message=s.get("message") or "",
                        calls=[c.get("function_name") for c in (s.get("tool_calls") or [])]))
    return out


def match(run_dir):
    log = next(iter(glob.glob(os.path.join(run_dir, "results", "vllm_requests.log*")) + glob.glob(os.path.join(run_dir, "vllm_requests.log*"))), None)
    if not log:
        sys.exit("no vllm_requests.log in " + run_dir)
    reqs = read_log(log)
    with_prompt = {k: v for k, v in reqs.items() if v.get("prompt_ids")}
    # a step is identified by its two recorded token counts (prompt, completion), which the log gives exactly as the lengths of
    # prompt_token_ids and output_token_ids; when two requests share both counts, the decoded output must begin the step's thought
    by_pair = collections.defaultdict(list)
    for k, v in with_prompt.items():
        by_pair[(len(v["prompt_ids"]), len(v["output_ids"]) if v.get("output_ids") is not None else None)].append(k)
    tok = None
    rows, stat = [], collections.Counter()
    used = set()
    for f in sorted(glob.glob(os.path.join(run_dir, "results", "traces", "trace_*.json"))):
        iid = os.path.basename(f)[6:-5]
        tr = json.load(open(f, encoding="utf-8"))
        for st in steps_of(tr):
            stat["agent steps"] += 1
            good = [k for k in by_pair.get((st["prompt_tokens"], st["completion_tokens"]), []) if k not in used]
            if len(good) > 1 and st["message"]:
                if tok is None:
                    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
                    from sne.runner import tokenizer
                    tok = tokenizer()
                key = st["message"].strip()[:60]
                good = [k for k in good if key and key in tok.decode(with_prompt[k]["output_ids"] or [])] or good
            if len(good) == 1:
                used.add(good[0])
                v = with_prompt[good[0]]
                rows.append(dict(iid=iid, step=st["step"], request=good[0], prompt_ids=v["prompt_ids"], output_ids=v.get("output_ids"),
                                 output_text=v.get("output_text"), calls=st["calls"], completion_tokens=st["completion_tokens"]))
                stat["matched to exactly one request"] += 1
            else:
                stat["no request" if not good else "several requests"] += 1
    stat["requests logged"] = len(reqs)
    stat["requests with prompt ids"] = len(with_prompt)
    stat["requests with output ids"] = sum(v.get("output_ids") is not None for v in reqs.values())
    stat["requests streamed"] = sum(bool(v.get("streamed")) for v in reqs.values())
    stat["requests not used by any step (sub-agent, summaries)"] = len(with_prompt) - len(used)
    return rows, stat, reqs


if __name__ == "__main__":
    run = sys.argv[1]
    rows, stat, reqs = match(run)
    for k, v in stat.items():
        print(f"{k}: {v}")
    if "--write" in sys.argv:
        with gzip.open(os.path.join(run, "served.jsonl.gz"), "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", os.path.join(run, "served.jsonl.gz"), len(rows), "rows")

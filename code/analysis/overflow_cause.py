"""What pushes a run over the context limit. The official server holds 32,768 tokens and the sample submission reserves 16,384 of them
for the answer, so a request fails once its prompt passes 16,384 tokens; history compaction starts only at 14,336. For every run of an
official Colab run that ended in a context overflow, this prints the prompt size of the last request that was served, the length of
the answer to it, and the tool results that were appended after it (the text that the failing request had to carry).
Usage: python tools/analysis/overflow_cause.py <run id> [more run ids]      e.g. EXP-138A EXP-014D"""
import ast
import collections
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
LIMIT, COMPACT = int(os.environ.get("SNE_LIMIT", 16384)), 14336


def lit(v):
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except Exception:
        try:
            return ast.literal_eval(v)
        except Exception:
            return v


def run(rid, only=None):
    d = os.path.join("artifacts_local", "colab", rid.lower(), "results")
    res = {}
    for l in open(os.path.join(d, "task_results.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        res[r["instance_id"]] = r
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "traces", "trace_*.json"))):
        t = os.path.basename(f)[6:-5]
        if only and t not in only:
            continue
        if "ContextWindowExceeded" not in str(res.get(t, {}).get("error")):
            continue
        last, tail, peak, calls = None, [], 0, 0
        for s in json.load(open(f, encoding="utf-8"))["steps"]:
            if lit(s.get("source")) != "agent":
                continue
            m = lit(s.get("metrics"))
            if isinstance(m, dict) and m.get("prompt_tokens"):
                last, tail = m, []
                peak = max(peak, m["prompt_tokens"])
                calls += 1
            ob = lit(s.get("observation"))
            tc = lit(s.get("tool_calls")) or []
            if ob and isinstance(tc, list):
                text = ob.get("content") if isinstance(ob, dict) else str(ob)
                for c in tc:
                    a = c.get("arguments") or {}
                    tail.append((c.get("function_name"), json.dumps(a)[:110], len(str(text)) // max(1, len(tc))))
        rows.append(dict(t=t, p=last["prompt_tokens"] if last else 0, c=last["completion_tokens"] if last else 0, tail=tail, peak=peak, calls=calls))
    return rows, res


for rid in sys.argv[1:]:
    rows, res = run(rid.upper())
    print(f"\n{rid.upper()}: {len(res)} runs, {len(rows)} ended in a context overflow")
    if not rows:
        continue
    kinds = collections.Counter()
    for x in sorted(rows, key=lambda x: -sum(n for _, _, n in x["tail"])):
        chars = sum(n for _, _, n in x["tail"])
        need = LIMIT - x["p"] - x["c"]  # tokens that were still free after the last served request and its answer
        if x["c"] >= 4000:
            k = "answer at the thinking cap or longer"
        elif chars / 3.0 >= max(need, 1):
            k = "tool result larger than the room left"
        elif x["p"] >= COMPACT:
            k = "already past the compaction threshold"
        else:
            k = "other"
        kinds[k] += 1
        big = max(x["tail"], key=lambda y: y[2]) if x["tail"] else ("-", "", 0)
        print(f"  {x['t']:18s} last served prompt {x['p']:6d} tokens, answer {x['c']:5d}, room left {need:6d}; tool results after it "
              f"{chars:7d} characters; largest: {big[0]} {big[1]} ({big[2]} characters) | {k}")
    print("  causes:", dict(kinds))
    print("  last served prompt, median", int(statistics.median(x["p"] for x in rows)), "| tool-result characters after it, median",
          int(statistics.median(sum(n for _, _, n in x["tail"]) for x in rows)))

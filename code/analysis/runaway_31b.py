"""31B's failures on the tasks the reference fix resolves: runaway runs (a response at the thinking cap, or 60+ tool calls) against
normal runs, from the saved traces of an official run (default EXP-074c, unedited 31B, seed 1) and its Kaggle scores.
Usage: python tools/analysis/runaway_31b.py [run id, default EXP-074C] [trace folder, default artifacts_local/colab/exp-074c]"""
import ast
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
RUN = (sys.argv[1] if len(sys.argv) > 1 else "EXP-074C").upper()
DIR = sys.argv[2] if len(sys.argv) > 2 else "artifacts_local/colab/exp-074c"
CAP, MANY = 4000, 60
off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"]][x["iid"]] = bool(x["resolved"])
gold = {t for t, v in off["GOLD"].items() if v}
res = {}
for l in open(os.path.join(DIR, "results", "task_results.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    res[r["instance_id"]] = r


def lit(v):
    try:
        return ast.literal_eval(v) if isinstance(v, str) else (v or {})
    except Exception:
        return {}


rows = []
for f in sorted(glob.glob(os.path.join(DIR, "results", "traces", "trace_*.json"))):
    t = os.path.basename(f)[6:-5]
    calls, ct = collections.Counter(), []
    for s in json.load(open(f, encoding="utf-8"))["steps"]:
        if s["source"] != "agent":
            continue
        for c in lit(s.get("tool_calls")) or []:
            calls[c.get("function_name")] += 1
        m = lit(s.get("metrics"))
        if m:
            ct.append(m.get("completion_tokens", 0))
    n = sum(calls.values())
    rows.append(dict(t=t, gold=t in gold, solved=bool(off[RUN].get(t)), calls=n, edits=calls["edit_file"], capped=max(ct or [0]) >= CAP,
                     many=n >= MANY, overflow="ContextWindowExceeded" in str(res.get(t, {}).get("error"))))
G = [x for x in rows if x["gold"]]
print(f"{RUN}: reference-resolvable tasks {len(G)}, solved {sum(x['solved'] for x in G)}")
for name, f in ((f"a response at the thinking cap (>= {CAP} tokens)", lambda x: x["capped"]), (f"{MANY}+ tool calls", lambda x: x["many"]),
                ("runaway (either)", lambda x: x["capped"] or x["many"]), ("normal (neither)", lambda x: not (x["capped"] or x["many"])),
                ("context overflow", lambda x: x["overflow"])):
    a = [x for x in G if f(x)]
    print(f"  {name:45s} tasks {len(a):2d}  solved {sum(x['solved'] for x in a):2d}")
print("unsolved runaway tasks:", ", ".join(f"{x['t']} ({x['calls']} calls, {x['edits']} edits{', overflow' if x['overflow'] else ''})"
                                           for x in G if not x["solved"] and (x["capped"] or x["many"])))

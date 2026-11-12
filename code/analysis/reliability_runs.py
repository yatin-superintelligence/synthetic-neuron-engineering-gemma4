"""Mechanism counts for every run on the 12-task reliability benchmark (EXP-114 onward): official solves when scored, edit calls and
how many applied, tool calls, no-patch runs and context errors, from each run's traces and task_results.jsonl."""
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"]][x["iid"]] = bool(x["resolved"])
cfg = {}
for f in glob.glob("experiments/colab_configs/*.json"):
    try:
        for e in json.load(open(f, encoding="utf-8")):
            if isinstance(e, dict) and "id" in e:
                cfg[e["id"].upper()] = e
    except Exception:  # noqa: BLE001
        pass
pref = sys.argv[1:] or ["exp-114", "exp-115", "exp-119", "exp-121"]
for d in sorted(glob.glob("artifacts_local/colab/exp-1[1-9][0-9]*")):
    rid = os.path.basename(d).upper()
    if not any(os.path.basename(d).startswith(p) for p in pref) or not os.path.exists(os.path.join(d, "results", "task_results.jsonl")):
        continue
    tr = [json.loads(l) for l in open(os.path.join(d, "results", "task_results.jsonl"), encoding="utf-8")]
    calls = edits = applied = 0
    for f in glob.glob(os.path.join(d, "results", "traces", "trace_*.json")):
        t = json.load(open(f, encoding="utf-8"))
        for s in t["steps"]:
            for c in (s.get("tool_calls") or []):
                calls += 1
                if c.get("function_name") == "edit_file":
                    edits += 1
                    o = json.dumps(s.get("observation") or {})
                    applied += not (('"status\\": \\"error' in o) or ('"status": "error' in o))
    solved = sum(off[rid].values()) if rid in off else None
    e = cfg.get(rid, {})
    title = e.get("title", "")[:60]
    print(f"{rid:9s} official {str(solved) + '/' + str(len(off[rid])) if solved is not None else 'pending':8s} | no patch {sum(x.get('agent_patch_size', 0) == 0 for x in tr):2d} "
          f"| context {sum('ContextWindow' in (x.get('error') or '') for x in tr):2d} | tool calls {calls:4d} | edits {applied}/{edits} applied | {title}")

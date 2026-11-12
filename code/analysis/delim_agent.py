"""Reads the 31B delimiter-neuron agent runs against their unedited baseline (D-172, D-176): per run, on the 40 reference-resolvable
tasks, how edit_file calls were answered (ok, old_string not found, refused for a missing parameter, other errors), how runs ended,
tool calls per run, minutes per task, and, from vLLM's request log when present, how many generated tool calls end a string with
backtick-comma before an argument name (the fault) and how many edit calls carry an old_string of 12 characters or fewer (a sign of
a string closed too early). Verifier scores are added when they exist.
Usage: python tools/analysis/delim_agent.py EXP-139B EXP-140A EXP-140D"""
import collections
import glob
import gzip
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "analysis"))
from _edits import applied  # noqa: E402

off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"]][x["iid"]] = bool(x["resolved"])
gold = {t for t, v in off["GOLD"].items() if v}
FAULT = re.compile(r"`,(old_string|new_string|filepath|allow_multiple):")


def run(rid):
    d = os.path.join("artifacts_local", "colab", rid.lower(), "results")
    R = {}
    for l in open(os.path.join(d, "task_results.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        R[r["instance_id"]] = r
    G = [t for t in R if t in gold]
    ans = collections.Counter()
    for t in G:
        f = os.path.join(d, "traces", f"trace_{t}.json")
        if not os.path.exists(f):
            continue
        for s in json.load(open(f, encoding="utf-8"))["steps"]:
            for c in (s.get("tool_calls") or []):
                if c.get("function_name") != "edit_file":
                    continue
                o = json.dumps(s.get("observation") or {})
                ans["calls"] += 1
                if applied(o):
                    ans["ok"] += 1
                elif "mandatory input parameters" in o:
                    ans["refused"] += 1
                elif "old_string not found" in o:
                    ans["not found"] += 1
                else:
                    ans["other error"] += 1
    end = collections.Counter(("overflow" if "ContextWindow" in str(R[t].get("error")) else "tool budget" if "tool call budget" in str(R[t].get("error"))
                               else "time" if "timeout" in str(R[t].get("error")) else "other error" if R[t].get("error") else "normal") for t in G)
    solved = {t for t in G if off[rid.upper()].get(t)}
    mins = sorted(R[t]["duration_seconds"] / 60 for t in G)
    smins = sorted(R[t]["duration_seconds"] / 60 for t in solved)
    calls = sum(R[t].get("tool_calls", 0) for t in G)
    out = dict(run=rid.upper(), tasks=len(G), solved=len(solved) if off[rid.upper()] else "not scored", edit_calls=dict(ans), ends=dict(end),
               tool_calls_per_run=round(calls / max(1, len(G)), 1), minutes_median=round(mins[len(mins) // 2], 1),
               minutes_per_solved_median=round(smins[len(smins) // 2], 1) if smins else None,
               solved_within=dict((m, sum(1 for t in solved if R[t]["duration_seconds"] <= m * 60)) for m in (4.5, 10, 20)))
    lg = os.path.join(d, "vllm_requests.log.gz")
    if os.path.exists(lg):
        fault = short = edits = 0
        with gzip.open(lg, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if " Generated response " not in line or "edit_file(" not in line:
                    continue
                edits += 1
                body = line[line.find("edit_file("):]
                fault += bool(FAULT.search(body)) or '"old_string"' not in body
                m = re.search(r'"old_string": "((?:[^"\\]|\\.)*)"', body)
                short += bool(m) and len(m.group(1)) <= 12
        out["request_log"] = dict(edit_calls=edits, faulty_string_end_or_no_old_string=fault, old_string_12_chars_or_fewer=short)
    return out, solved


rows = [run(r) for r in sys.argv[1:]]
for o, _ in rows:
    print(json.dumps(o))
if len(rows) >= 2:
    base = rows[0][1]
    for o, s in rows[1:]:
        if isinstance(o["solved"], int):
            print(f"{o['run']} against {rows[0][0]['run']}: gained {len(s - base)} {sorted(s - base)}, lost {len(base - s)} {sorted(base - s)}")

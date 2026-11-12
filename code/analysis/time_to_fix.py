"""When a solved run applied its final source edit, against the competition's time budget (about 4.5 minutes per task for 31B within
the 12-hour limit, D-076), and what the run did before it. Times are the harness's elapsed seconds in a run at 16 tasks at once, so
they are a proxy for the hidden test's one-task-at-a-time timing. Edits to files whose path contains "test" or "repro" are not counted.
Usage: python tools/analysis/time_to_fix.py RUN_ID run_folder [cap seconds, default 270]   e.g. EXP-074C artifacts_local/colab/exp-074c"""
import ast
import collections
import glob
import json
import statistics
import sys

run, d = sys.argv[1].upper(), sys.argv[2]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 270
off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"]][x["iid"]] = bool(x["resolved"])


def lit(v):
    try:
        return ast.literal_eval(v) if isinstance(v, str) else (v or {})
    except Exception:
        return {}


R = []
for t, ok in off[run].items():
    if not ok:
        continue
    steps = [s for s in json.load(open(f"{d}/results/traces/trace_{t}.json", encoding="utf-8"))["steps"] if s["source"] == "agent"]
    last = None
    for k, s in enumerate(steps):
        for c in lit(s.get("tool_calls")) or []:
            a = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
            name = a.get("filepath", "")
            applied = '"status": "ok"' in json.dumps(s.get("observation") or "").replace('\\"', '"')
            if c.get("function_name") in ("edit_file", "write_file") and applied and "test" not in name and "repro" not in name:
                last = k
    if last is None:
        continue
    pre = steps[:last + 1]
    kinds = collections.Counter(c.get("function_name") for s in pre for c in (lit(s.get("tool_calls")) or []))
    R.append(dict(calls=len(pre), tokens=sum(lit(s.get("metrics")).get("completion_tokens") or 0 for s in pre), elapsed=lit(pre[-1].get("extra")).get("elapsed_s") or 0,
                  reads=kinds["read_file"], commands=kinds["run_command"], edits=kinds["edit_file"] + kinds["write_file"]))
print(f"{run}: solved runs with a source edit {len(R)}; final source edit within {cap} s: {sum(r['elapsed'] <= cap for r in R)}; median {int(statistics.median(r['elapsed'] for r in R))} s")
for name, g in ((f"within {cap} s", [r for r in R if r["elapsed"] <= cap]), (f"after {cap} s", [r for r in R if r["elapsed"] > cap])):
    if g:
        m = lambda k: statistics.median(r[k] for r in g)  # noqa: E731
        print(f"  {name}: {len(g)} runs; before the final edit, medians: model calls {m('calls')}, generated tokens {int(m('tokens'))}, reads {m('reads')}, commands {m('commands')}, edit and write calls {m('edits')}")

"""Where E2B's submitted patches land relative to the reference fix, over every official E2B run with a Kaggle score.

For each task-run with a submitted patch: solved (official), else right lines (a changed line of the original file within 3 lines
of a line the reference fix changes), right file but other lines, or wrong file only. Also per task: how often any run reaches the
right lines. Separates two bottlenecks: finding the place (localization) and writing the right change there (fix content).
Usage: python tools/analysis/localization.py [model prefix, default gemma-4-e2b]
"""
import collections
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-e2b"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def changed(patch):
    """{file: set of original-file line numbers removed or next to an insertion}"""
    out, f, ln = collections.defaultdict(set), None, 0
    for line in (patch or "").splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            f = line[4:].strip().removeprefix("b/")
            continue
        m = HUNK.match(line)
        if m:
            ln = int(m.group(1))
            continue
        if f is None or line.startswith("\\"):
            continue
        if line.startswith("-"):
            out[f].add(ln)
            ln += 1
        elif line.startswith("+"):
            out[f].add(ln)
        else:
            ln += 1
    return out


ref = {}
for l in open("data/kaggle/tasks.jsonl", encoding="utf-8"):
    t = json.loads(l)
    ref[t["instance_id"]] = changed(t["patch"])
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
runs = sorted(r for r in off if cfg.get(r.upper(), {}).get("model", "").startswith(PREFIX) and len(off[r]) >= 56)
tot = collections.Counter()
per = collections.defaultdict(collections.Counter)
for r in runs:
    d = next((os.path.join(b, r.lower(), "results", "patches") for b in ("artifacts_local/colab", "artifacts_local/kaggle")
              if os.path.isdir(os.path.join(b, r.lower(), "results", "patches"))), None)
    if d is None:
        continue
    for iid, ok in off[r].items():
        p = os.path.join(d, f"{iid}.patch")
        txt = open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
        if ok:
            k = "solved"
        elif not txt.strip():
            k = "no patch"
        else:
            a, g = changed(txt), ref.get(iid, {})
            files = set(a) & set(g)
            if not files:
                k = "wrong file only"
            elif any(abs(x - y) <= 3 for f in files for x in a[f] for y in g[f]):
                k = "right lines, wrong change"
            else:
                k = "right file, other lines"
        tot[k] += 1
        per[iid][k] += 1
n = sum(tot.values())
print(f"{PREFIX}: {len(runs)} runs, {n} task-runs")
for k, v in tot.most_common():
    print(f"  {k:28s} {v:5d}  {100 * v / n:5.1f}%")
reach = {i: c["solved"] + c["right lines, wrong change"] for i, c in per.items()}
runs_per = {i: sum(c.values()) for i, c in per.items()}
ever = [i for i in per if reach[i] > 0]
print(f"tasks where some run reached the right lines: {len(ever)} of {len(per)}")
print("tasks never solved but reached at the right lines (runs reaching / runs):")
for i in sorted(per, key=lambda i: -reach[i]):
    if per[i]["solved"] == 0 and reach[i] > 0:
        print(f"  {i:22s} {reach[i]:3d} / {runs_per[i]}")
print("solved tasks (solved / right lines wrong change / runs):")
for i in sorted(per, key=lambda i: -per[i]["solved"]):
    if per[i]["solved"]:
        print(f"  {i:22s} {per[i]['solved']:3d} / {per[i]['right lines, wrong change']:3d} / {runs_per[i]}")

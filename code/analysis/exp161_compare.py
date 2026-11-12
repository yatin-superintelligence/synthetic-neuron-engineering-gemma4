"""EXP-161a per-task comparison (D-223): the causal-neuron edit against the six unedited 31B runs at the 8,192 setting and the six
correlation-chosen edits, on the 15 tasks 31B solves only sometimes. All scores are the official verification on Kaggle.

Unedited (8,192, no LoRA or an all-zero LoRA): EXP-139B (seed 1), 139C (2), 139D (3), 140E (zero adapter, 1), 150B (4), 150D (5).
Correlation edits: EXP-155A (forward), EXP-156R0 (reverse), 156R1 (gentle), 156R2 (amplify-only), 157R0 (damp-only), 157R1 (top-16).
Usage: python tools/analysis/exp161_compare.py"""
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
UNEDITED = ("EXP-139B", "EXP-139C", "EXP-139D", "EXP-140E", "EXP-150B", "EXP-150D")
CORR = ("EXP-155A", "EXP-156R0", "EXP-156R1", "EXP-156R2", "EXP-157R0", "EXP-157R1")
RUN = sys.argv[1] if len(sys.argv) > 1 else "EXP-161A"

off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"].upper()][x["iid"].split("__r")[0]] = bool(x["resolved"])  # round copies carry __r<round><arm>
tasks = [json.loads(l)["instance_id"] for l in open("artifacts_local/colab/exp-161a/results/task_results.jsonl", encoding="utf-8")]
corr = [r for r in CORR if off.get(r)]
print("runs found: unedited", [r for r in UNEDITED if off.get(r)], "| correlation edits", corr, "| this run", RUN in off)
print(f"{'task':16s} {'unedited':>9s} {'corr':>7s} {RUN:>9s}")
tot = [0, 0, 0, 0, 0]
for t in sorted(tasks):
    u = sum(off[r].get(t, 0) for r in UNEDITED)
    c = sum(off[r].get(t, 0) for r in corr)
    cn = sum(1 for r in corr if t in off[r])
    e = off[RUN].get(t)
    tot[0] += u; tot[1] += c; tot[2] += cn; tot[3] += bool(e); tot[4] += e is not None
    mark = "  gained vs mostly-failing" if e and u <= 2 else ("  lost vs mostly-solving" if e is False and u >= 5 else "")
    print(f"{t:16s} {u:>5d} of 6 {c:>3d} of {cn} {str(e):>9s}{mark}")
print(f"\nunedited {tot[0]} of 90 (per run {[sum(off[r].get(t, 0) for t in tasks) for r in UNEDITED]}) | correlation edits {tot[1]} of {tot[2]}"
      f" (per run {[sum(off[r].get(t, 0) for t in tasks) for r in corr]}) | {RUN} {tot[3]} of {tot[4]}")

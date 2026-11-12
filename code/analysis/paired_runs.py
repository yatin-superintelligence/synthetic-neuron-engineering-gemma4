"""Paired comparison of two arms on the reliability benchmark by (task, seed): python tools/analysis/paired_runs.py A1,A2 B1,B2
(run ids at matched seeds, in the same order). Prints solves of each arm, discordant pairs and the exact two-sided sign test."""
import collections
import glob
import json
import math
import sys

off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"]][x["iid"]] = bool(x["resolved"])
A, B = sys.argv[1].upper().split(","), sys.argv[2].upper().split(",")
a = b = x = y = n = 0
for ra, rb in zip(A, B):
    for t in sorted(set(off[ra]) | set(off[rb])):
        va, vb = off[ra].get(t, False), off[rb].get(t, False)  # a task without a score (cut by the time limit) counts as unsolved
        n += 1
        a += va
        b += vb
        x += va and not vb
        y += vb and not va
        if va != vb:
            print(f"  {t:18s} {ra}={int(va)} {rb}={int(vb)}")
k = x + y
p = 1.0 if k == 0 else min(1.0, 2 * sum(math.comb(k, i) for i in range(min(x, y) + 1)) / 2 ** k)
print(f"pairs {n}: {'+'.join(A)} solved {a}, {'+'.join(B)} solved {b}; discordant {x} vs {y}, exact sign test p = {p:.3f}")

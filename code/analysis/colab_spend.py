"""Where the Colab units went: every runtime session in experiments/_system/colab.log (batch workers, script jobs, attached drivers),
its GPU, hours and cost at colab/BUDGET.md's rates, and whether it produced output (a recorded run or unpacked results).
Usage: python tools/analysis/colab_spend.py"""
import collections
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RATE = {"T4": 1.15, "L4": 1.94, "A100": 6.73, "G4": 10.05}
TS = re.compile(r"^(\d{4}-\d\d-\d\dT[\d:]+Z) (\S+) (.*)$")
open_ = {}
sessions = []
for line in open(os.path.join(ROOT, "experiments", "_system", "colab.log"), encoding="utf-8", errors="replace"):
    m = TS.match(line.rstrip("\n"))
    if not m:
        continue
    t, tag, rest = m.groups()
    s = open_.get(tag)
    g = re.match(r"start (T4|L4|A100|G4) session", rest)
    j = re.match(r"job start (\S+) (T4|L4|A100|G4) ", rest)
    if g or j or rest.startswith("attached to "):
        gpu = (g.group(1) if g else j.group(2) if j else (s or {}).get("gpu", "L4"))
        open_[tag] = dict(tag=tag, gpu=gpu, start=t, what=(j.group(1).split("/")[-1] if j else "batch worker" if g else "attached driver"),
                          recorded=0, failed=0, unpacked=False, notes=[])
        continue
    if s is None:
        continue
    if " RECORDED " in f" {rest} ":
        s["recorded"] += 1
    if "FAILED" in rest or "time limit" in rest:
        s["failed"] += 1
    if "outputs unpacked" in rest:
        s["unpacked"] = True
    e = re.match(r"(?:job )?end (?:\(attached driver\) )?([\d.]+) h", rest)
    if e:
        s.update(end=t, hours=float(e.group(1)))
        s["cost"] = round(s["hours"] * RATE[s["gpu"]], 2)
        s["output"] = s["recorded"] > 0 or s["unpacked"]
        sessions.append(s)
        open_.pop(tag, None)
tot = sum(s["cost"] for s in sessions)
none = [s for s in sessions if not s["output"]]
print(f"sessions {len(sessions)}, cost {tot:.1f} units at BUDGET.md rates (log-based; the account also bills idle seconds between log lines)")
print(f"sessions with no output: {len(none)}, {sum(s['cost'] for s in none):.1f} units")
by = collections.defaultdict(float)
for s in none:
    by[s["gpu"]] += s["cost"]
print("  by GPU:", {k: round(v, 1) for k, v in by.items()})
print("  largest:")
for s in sorted(none, key=lambda s: -s["cost"])[:15]:
    print(f"   {s['start'][:16]} {s['tag']:9s} {s['gpu']:4s} {s['hours']:5.2f} h {s['cost']:6.2f} u  {s['what']}  failed-lines {s['failed']}")
small = [s for s in none if s["hours"] < 0.1]
print(f"  of which ended within 6 minutes (allocation outages, early errors): {len(small)}, {sum(s['cost'] for s in small):.1f} units")
print("by GPU, all sessions:", {g: round(sum(s['cost'] for s in sessions if s['gpu'] == g), 1) for g in RATE})

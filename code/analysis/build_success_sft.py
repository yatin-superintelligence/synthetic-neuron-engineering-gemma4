"""Builds EXP-110's training data: every officially verified successful agent run (any model, unedited or edited) as a chat-template
conversation, split by the shared dev/test halves of the 58 tasks (sne.loops.split_map), so each half's model is trained only on the
other half's tasks. Output (private, git-ignored): artifacts_local/EXP-110/success_{dev,test}.jsonl with {run, model, iid, messages}."""
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "src")
from sne.loops import messages_before, split_map  # noqa: E402

off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"]][x["iid"]] = bool(x["resolved"])
split = split_map()


def model_of(run):
    d = glob.glob(f"experiments/EXP-{run[4:]}_*") + glob.glob(f"experiments/EXP-{run[4:].lower()}_*") + glob.glob(f"experiments/EXP-{run[4:].upper()}_*")
    if not d:
        return "?", None
    c = os.path.join(d[0], "config.json")
    txt = (open(c, encoding="utf-8").read() if os.path.exists(c) else "") + d[0]
    txt = txt.lower()
    return ("31b" if "31b" in txt else "12b" if "12b" in txt else "e4b" if "e4b" in txt else "e2b" if "e2b" in txt else "?"), d[0]


def trace_dir(run):
    for base in ("artifacts_local/colab", "artifacts_local/kaggle"):
        p = os.path.join(base, run.lower(), "results", "traces")
        if os.path.isdir(p):
            return p
    return None


out = {"dev": [], "test": []}
stats = collections.Counter()
for run, res in sorted(off.items()):
    if run == "GOLD" or run.startswith("SMOKE") or run.endswith("partial"):
        continue
    m, d = model_of(run)
    td = trace_dir(run)
    if td is None:
        stats["no traces"] += 1
        continue
    for iid, ok in res.items():
        if not ok:
            continue
        f = os.path.join(td, f"trace_{iid}.json")
        if not os.path.exists(f):
            stats["missing trace"] += 1
            continue
        t = json.load(open(f, encoding="utf-8"))
        msgs = messages_before(t, len(t["steps"]))
        if not any(mm.get("role") == "assistant" for mm in msgs):
            continue
        names = {str(s_.get("model_name") or "").lower() for s_ in t["steps"] if s_.get("source") == "agent"}
        mm_ = "31b" if any("31b" in n for n in names) else "12b" if any("12b" in n for n in names) else "e4b" if any("e4b" in n for n in names)             else "e2b" if any("e2b" in n for n in names) else m  # the model that actually ran, from the trace
        m = mm_
        half = split.get(iid)
        out[half].append(dict(run=run, model=m, iid=iid, messages=msgs))
        stats[(half, m)] += 1
os.makedirs("artifacts_local/EXP-110", exist_ok=True)
for h, rows in out.items():
    with open(f"artifacts_local/EXP-110/success_{h}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
print({str(k): v for k, v in sorted(stats.items(), key=str)})
for h, rows in out.items():
    print(h, "trajectories", len(rows), "distinct tasks", len({r["iid"] for r in rows}), "by model", dict(collections.Counter(r["model"] for r in rows)))

"""Bench for EXP-147 (D-202): the state where 31B first edits a file that the task's reference fix touches, in its own recorded runs of
the tasks it solves only sometimes, labelled by whether that run's patch passed the official verification.

Why: 31B's remaining failures on solvable tasks are wrong fixes (13 of 93 attempts at the 8,192 setting), and on the 15 tasks it
solves in 2 to 9 of 10 unedited runs the knowledge is in the model: failing is a choice. If the activations at the first real edit
separate runs that will pass from runs that will not, across tasks (held out by task), there is a direction an edit can push.

Sources: every 31B run with a vLLM request log (results/vllm_requests.log.gz) and a verifier score. Edited runs (the delimiter and
gated neurons and their controls) are kept: their requests are used as token sequences on the unedited model, and their trajectories
were judged by the same verifier. Replica copies (<task>__r<k>) count as separate runs of their task.
A row: run, iid (the copy's id), task (the base id), step, solved (verifier), prompt_ids, output_ids, edit file.
Usage: python tools/analysis/build_success_bench.py   (writes artifacts_local/EXP-147/success_bench.jsonl.gz)"""
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
import served_prompts as SP  # noqa: E402

RUNS = ("EXP-129A", "EXP-131A", "EXP-139B", "EXP-139D", "EXP-140A", "EXP-140D", "EXP-140E", "EXP-143C", "EXP-144A", "EXP-144B", "EXP-145A")
UNEDITED = ("EXP-074C", "EXP-129A", "EXP-129B", "EXP-131A", "EXP-139B", "EXP-139C", "EXP-139D", "EXP-140E", "EXP-143A", "EXP-143C")
if "--model" in sys.argv and sys.argv[sys.argv.index("--model") + 1] == "12b":  # EXP-158: the same bench for 12B
    RUNS = ("EXP-138A", "EXP-139A", "EXP-141E", "EXP-159A", "EXP-159B")  # EXP-133a, 137a: old log format; EXP-159c dropped (D-218)
    UNEDITED = ("EXP-014D", "EXP-138A", "EXP-139A", "EXP-141A", "EXP-141B2", "EXP-141C", "EXP-141E", "EXP-141F", "EXP-141G", "EXP-148A", "EXP-148B", "EXP-153A", "EXP-153B")
POINT = sys.argv[sys.argv.index("--point") + 1] if "--point" in sys.argv else "first_edit"  # first_edit (EXP-147) or submit (EXP-152)
OUT = os.path.join("artifacts_local", "EXP-152", "submit_bench.jsonl.gz") if POINT == "submit" else os.path.join("artifacts_local", "EXP-147", "success_bench.jsonl.gz")
if "--model" in sys.argv and sys.argv[sys.argv.index("--model") + 1] == "12b":
    OUT = os.path.join("artifacts_local", "EXP-158", "success_bench.jsonl.gz")


def main():
    off = collections.defaultdict(dict)
    for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
        for x in json.load(open(f, encoding="utf-8")):
            off[x["run"].upper()][x["iid"]] = bool(x["resolved"])
    gold = sorted(t for t, v in off["GOLD"].items() if v)
    some = [t for t in gold if 0 < sum(off[r].get(t, 0) for r in UNEDITED) < sum(1 for r in UNEDITED if t in off[r])]
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open("data/kaggle/tasks.jsonl", encoding="utf-8")}
    files = {t: set(re.findall(r"^\+\+\+ b/(\S+)", tasks[t]["patch"], re.M)) for t in some}
    rows, miss = [], collections.Counter()
    for run in RUNS:
        d = os.path.join("artifacts_local", "colab", run.lower())
        matched, stat, _ = SP.match(d)
        by = collections.defaultdict(list)
        for r in matched:
            by[r["iid"]].append(r)
        for iid, rs in by.items():
            base = iid.split("__r")[0]
            if base not in some or iid not in off[run]:
                continue
            tr = json.load(open(os.path.join(d, "results", "traces", f"trace_{iid}.json"), encoding="utf-8"))["steps"]
            first = None
            if POINT == "submit":  # the last request whose output calls submit_patch
                for r in sorted(rs, key=lambda r: -r["step"]):
                    if any(c.get("function_name") == "submit_patch" for c in (tr[r["step"]].get("tool_calls") or [])):
                        first = (r, "submit_patch")
                        break
            for r in ([] if POINT == "submit" else sorted(rs, key=lambda r: r["step"])):
                for c in (tr[r["step"]].get("tool_calls") or []):
                    a = c.get("arguments") or {}
                    if isinstance(a, str):
                        try:
                            a = json.loads(a)
                        except Exception:  # noqa: BLE001
                            a = {}
                    fp = str(a.get("filepath") or a.get("path") or "")
                    if c.get("function_name") in ("edit_file", "write_file") and fp in files[base]:
                        first = (r, fp)
                        break
                if first:
                    break
            if not first or not first[0].get("output_ids"):
                miss[run] += 1
                continue
            r, fp = first
            rows.append(dict(run=run, iid=iid, task=base, step=r["step"], solved=bool(off[run][iid]), file=fp,
                             prompt_ids=r["prompt_ids"], output_ids=r["output_ids"]))
        print(run, "steps matched", stat.get("matched to exactly one request"), "of", stat.get("agent steps"), "| rows so far", len(rows))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["task"]][0] += r["solved"]
        per[r["task"]][1] += 1
    print("\nrows", len(rows), "| solved", sum(r["solved"] for r in rows), "| runs without a first edit to a reference file:", dict(miss))
    for t in sorted(per):
        print(f"  {t:16s} solved {per[t][0]} of {per[t][1]}")
    usable = [t for t, (s, n) in per.items() if 0 < s < n]
    print("tasks with both outcomes:", len(usable), "| prompt tokens median", sorted(len(r["prompt_ids"]) for r in rows)[len(rows) // 2],
          "| wrote", OUT, round(os.path.getsize(OUT) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()

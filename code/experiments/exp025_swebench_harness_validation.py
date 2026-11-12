"""EXP-025: validate SWE-bench scoring on this machine before scoring any model.

For 12 dev instances (stratified over repos by a fixed rule) run score() with (a) the gold patch and (b) a no-op comment
patch. Gold must resolve; the no-op must not. Any failure is a harness problem, not a model result.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import sweagent as A  # noqa: E402
from sne.registry import Experiment  # noqa: E402

with Experiment("EXP-025", "swebench-harness-validation", category="benchmark", depends_on=["EXP-002"],
                motivation="The headline claim must rest on real issues. Scoring must first reproduce SWE-bench's own "
                           "ground truth here: gold patches pass, unrelated patches fail.",
                hypothesis="12/12 gold patches resolve and 0/12 no-op patches resolve.",
                config=dict(n=12, subset="dev", rule="first instances per repo in sorted id order, django capped at 4",
                            scoring="official eval_script + official log parser in a fresh container, no network")) as ex:
    A.ensure_docker(ex.log)
    dev = sorted(A.load_subset("dev"), key=lambda r: r["instance_id"])
    pick, cnt = [], collections.Counter()
    for r in dev:
        if cnt[r["repo"]] < (4 if r["repo"] == "django/django" else 3) and len(pick) < 12:
            pick.append(r)
            cnt[r["repo"]] += 1
    ex.log([r["instance_id"] for r in pick])
    A.pull(pick, ex.log)
    rows = []
    for r in pick:
        g = A.score(r, r["patch"])
        path = A.re.findall(r"^diff --git a/(\S+)", r["patch"], A.re.M)[0]
        noop = (f"diff --git a/NOOP.txt b/NOOP.txt\nnew file mode 100644\n--- /dev/null\n+++ b/NOOP.txt\n@@ -0,0 +1 @@\n+noop\n")
        n = A.score(r, noop)
        rows.append(dict(id=r["instance_id"], gold_path=path, gold={k: v for k, v in g.items() if k != "log_tail"},
                         noop={k: v for k, v in n.items() if k != "log_tail"}, gold_log=g.get("log_tail", "")[-500:]))
        ex.log(r["instance_id"], "gold", g["resolved"], f"{g['f2p_passed']}/{g['f2p_total']}", g.get("n_parsed"),
               "noop", n["resolved"], f"{n['f2p_passed']}/{n['f2p_total']}")
        ex.save_json("rows.json", rows)
    gold_ok = sum(x["gold"]["resolved"] for x in rows)
    noop_ok = sum(x["noop"]["resolved"] for x in rows)
    disk = A.subprocess.run(["docker", "system", "df"], capture_output=True, text=True).stdout
    ex.log(disk)
    ex.result(gold_resolved=gold_ok, noop_resolved=noop_ok, n=len(rows), docker_df=disk,
              headline=f"gold {gold_ok}/{len(rows)} resolved; no-op {noop_ok}/{len(rows)} resolved")
    A.stop_docker()

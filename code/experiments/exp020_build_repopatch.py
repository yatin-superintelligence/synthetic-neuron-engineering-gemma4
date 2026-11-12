"""EXP-020: build and validate RepoPatch v1 (our own agentic bug-fix benchmark in the competition tool format)."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import DATA, repobench  # noqa: E402
from sne.execute import run_programs  # noqa: E402
from sne.registry import Experiment  # noqa: E402

with Experiment("EXP-020", "build-repopatch-v1", category="benchmark", depends_on=["EXP-002"],
                motivation="The competition scores agents on resolving real issues by editing files through tool "
                           "calls. Function-completion benchmarks miss three skills a small agent needs: locating the "
                           "faulty code among distractors, reading an issue report, and emitting well-formed edit "
                           "calls whose old_string matches exactly. RepoPatch measures these, cheaply and executably.",
                hypothesis="Every kept task is valid: the oracle edit resolves it and the unedited repo fails.",
                config=dict(n_target=150, seed=repobench.SEED, mutation_ops=["cmp", "const", "arith", "bool", "not"],
                            files=["pkg/core.py", "pkg/utils.py"], distractors=4)) as ex:
    tasks = repobench.build(150)
    oracle = run_programs([(t["task_id"], repobench.hidden_program(t, repobench.apply_calls(
        t["files"], [dict(filepath="pkg/core.py", old_string=t["buggy_fn"], new_string=t["canonical_fn"])])))
        for t in tasks])
    buggy = run_programs([(t["task_id"], repobench.hidden_program(t, t["files"])) for t in tasks])
    ok = [t for t in tasks if oracle[t["task_id"]]["passed"] and not buggy[t["task_id"]]["passed"]]
    path = os.path.join(DATA, "repopatch_v1.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for t in ok:
            f.write(json.dumps({k: v for k, v in t.items() if k != "_probe"}) + "\n")
    muts = collections.Counter(t["mutation"] for t in ok)
    lens = [len(repobench.render_task(t)) for t in ok]
    ex.save_json("example_prompt.json", dict(task=ok[0]["task_id"], prompt=repobench.render_task(ok[0])))
    ex.result(n_tasks=len(ok), mutations=dict(muts), oracle_resolves=sum(oracle[t["task_id"]]["passed"] for t in tasks),
              buggy_passes=sum(buggy[t["task_id"]]["passed"] for t in tasks),
              prompt_chars=dict(min=min(lens), max=max(lens), mean=sum(lens) / len(lens)), path=path,
              headline=f"{len(ok)} valid tasks; mutations {dict(muts)}")

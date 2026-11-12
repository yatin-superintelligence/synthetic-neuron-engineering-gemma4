"""EXP-021: build and validate CopyBench v1 (src/sne/copybench.py). CPU only, seconds."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import copybench as C  # noqa: E402
from sne.registry import Experiment  # noqa: E402

with Experiment("EXP-021", "build-copybench-v1", category="benchmark", depends_on=[],
                motivation="H-1 needs a cheap, high-power measure of exact copying from context, the skill behind "
                           "edit_file old_string, paths and required identifiers. Real stdlib code keeps it disjoint "
                           "from every other benchmark. Short outputs allow screening many edits.",
                hypothesis="All oracle answers pass the judge and trivially wrong answers fail.",
                config=dict(seed=C.SEED, modules=C.MODULES, per_kind=100)) as ex:
    tasks = C.build()
    with open(C.PATH, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    oracle, wrong = collections.Counter(), collections.Counter()
    for t in tasks:
        if t["kind"] == "name":
            good, bad = f"```python\ndef {t['required']}(x):\n    return x\n```", "```python\ndef solve(x):\n    return x\n```"
        elif t["kind"] == "verbatim":
            good, bad = f"```python\n{t['expected']}\n```", f"```python\n{t['expected'].replace('    ', '   ')}x\n```"
        else:
            line = t["line"]
            new = [l for l, o in zip(t["expected"].split("\n"), t["body"].split("\n")) if l != o][0]
            q = '<|"|>'
            good = f"<|tool_call>call:edit_file{{filepath:{q}{t['path']}{q},old_string:{q}{line}{q},new_string:{q}{new}{q}}}<tool_call|>"
            bad = f"<|tool_call>call:edit_file{{filepath:{q}{t['path']}{q},old_string:{q}{line}zz{q},new_string:{q}{new}{q}}}<tool_call|>"
        oracle[t["kind"]] += C.judge(t, good)[0]
        wrong[t["kind"]] += C.judge(t, bad)[0]
    ex.result(n=len(tasks), kinds=dict(collections.Counter(t["kind"] for t in tasks)),
              splits=dict(collections.Counter(t["split"] for t in tasks)), oracle_pass=dict(oracle), wrong_pass=dict(wrong),
              headline=f"{len(tasks)} tasks; oracle pass {dict(oracle)}; wrong pass {dict(wrong)}")

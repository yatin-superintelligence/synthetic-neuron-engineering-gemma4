"""EXP-002: validate the evaluation harness before any model is scored.

Canonical solutions must pass 100%; HumanEvalFix buggy solutions must fail 100% on hidden tests;
canonical HumanEvalFix fixes must also pass the visible example tests. Any deviation is a harness bug.
Also checks code extraction on synthetic chat-style responses.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import bench  # noqa: E402
from sne.execute import run_programs  # noqa: E402
from sne.registry import Experiment  # noqa: E402

with Experiment("EXP-002", "harness-validation", category="infrastructure",
                motivation="Every later number depends on prompt assembly, code extraction and test execution being "
                           "correct. A reviewer should be able to see that the harness passes reference solutions "
                           "and rejects known-buggy ones before trusting any model score.",
                hypothesis="Canonical solutions pass 100% on all suites; buggy HumanEvalFix solutions fail 100%.",
                config=dict(suites=["mbpp_dev", "mbpp_test", "humaneval", "humanevalfix"], timeout_s=30)) as ex:
    out = {}
    for s in ["mbpp_dev", "mbpp_test", "humaneval", "humanevalfix"]:
        tasks = bench.get_suite(s)
        res = run_programs([(t["task_id"], bench.build_program(t, t["_canonical"])) for t in tasks])
        bad = [k for k, v in res.items() if not v["passed"]]
        out[s] = dict(n=len(tasks), canonical_pass=len(tasks) - len(bad), failures=bad[:20])
        ex.log(s, out[s])
        if s == "humanevalfix":
            rb = run_programs([(t["task_id"], bench.build_program(t, t["_buggy"])) for t in tasks])
            ok_buggy = [k for k, v in rb.items() if v["passed"]]
            rv = run_programs([(t["task_id"], bench.build_program(t, t["_canonical"], visible=True)) for t in tasks])
            out["humanevalfix_buggy_hidden"] = dict(n=len(tasks), buggy_unexpectedly_pass=ok_buggy)
            out["humanevalfix_canonical_visible"] = dict(fail=[k for k, v in rv.items() if not v["passed"]])
            rbv = run_programs([(t["task_id"], bench.build_program(t, t["_buggy"], visible=True)) for t in tasks])
            out["humanevalfix_buggy_visible_pass"] = sum(v["passed"] for v in rbv.values())
            ex.log("buggy hidden unexpectedly pass:", ok_buggy, "canonical visible fails:",
                   out["humanevalfix_canonical_visible"], "buggy passing visible examples:",
                   out["humanevalfix_buggy_visible_pass"])
    # extraction unit checks
    cases = [("Here:\n```python\ndef f(x):\n    return x\n```\nDone", "f", "def f(x):\n    return x\n"),
             ("```python\nimport math\n```\ntext\n```python\ndef g(a):\n    return a\n```", "g", "def g(a):\n    return a\n"),
             ("```python\ndef h():\n    return 1\n", "h", "def h():\n    return 1\n"),
             ("def k():\n    return 2", "k", "def k():\n    return 2")]
    ext = [bench.extract_code(t, e) == want for t, e, want in cases]
    out["extraction_unit_tests"] = ext
    ex.log("extraction", ext)
    ex.result(**out, headline=" | ".join(f"{s}: {out[s]['canonical_pass']}/{out[s]['n']}" for s in
                                         ["mbpp_dev", "mbpp_test", "humaneval", "humanevalfix"]))

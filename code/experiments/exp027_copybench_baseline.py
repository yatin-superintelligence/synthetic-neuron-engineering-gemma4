"""EXP-027: CopyBench baselines (all 300 tasks) for E2B F16 and Q4_0, with E4B and 12B Q4_0 if present.

First direct measurement of H-1. If E2B is weak at exact copying relative to larger siblings, copying is a candidate
removable obstacle. Per-kind and failure-reason breakdowns are the primary outputs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import MODELS  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402

GG = os.path.join(MODELS, "gguf")
SUITES = ["copybench"]
with Experiment("EXP-027", "copybench-baseline", category="baseline", depends_on=["EXP-021", "EXP-003"],
                motivation="Directly measures exact copying from context, the mechanism proposed by H-1.",
                hypothesis="E2B fails a meaningful share of copy tasks (below 90%), most on verbatim whitespace and on "
                           "required names; larger siblings do better.",
                config=dict(suites=SUITES, parallel=8, max_tokens=4096)) as ex:
    arms = [Arm("e2b_f16", "f16"), Arm("e2b_q4_0", "q4_0")]
    for n, f in [("e4b_q4_0", "gemma-4-E4B_q4_0-it.gguf"), ("12b_q4_0", "gemma-4-12b-it-qat-q4_0.gguf")]:
        if os.path.exists(os.path.join(GG, f)):
            arms.append(Arm(n, "q4_0", gguf=os.path.join(GG, f)))
    res, _ = evaluate_arms(ex, arms, SUITES, parallel=8)
    ex.result(by_arm={k: v["copybench"]["by_kind"] for k, v in res.items()},
              headline=" | ".join(f"{k}: {v['copybench']['pass_rate']:.3f} {v['copybench']['by_kind']}" for k, v in res.items()))

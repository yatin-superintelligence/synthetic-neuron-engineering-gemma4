"""EXP-017: Gemma 4 thinking mode on vs off, plus LiveCodeBench v6-sized baseline, F16 and Q4_0.

Questions: (1) does our LiveCodeBench harness reproduce Google's published 44.0% for E2B (coarse calibration; their
setup is unknown, thinking was likely on)? (2) Does thinking absorb the reasoning-in-comments behaviour (H-2), reduce
truncation and interface blindness on MBPP+ dev? Thinking is a free, existing behaviour switch, so it is a mandatory
baseline for any edit that targets reasoning or stopping.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402

SUITES = ["lcb"]
with Experiment("EXP-017", "thinking-mode-and-livecodebench", category="baseline", depends_on=["EXP-010"],
                motivation="External calibration against Google's number, and a free behaviour switch that every "
                           "reasoning-related edit must beat.",
                hypothesis="Thinking on raises LiveCodeBench into the low 40s and lowers MBPP+ truncation; thinking off "
                           "stays well below 44%.",
                config=dict(suites=SUITES, parallel=8, ctx_per_slot=16384, max_tokens=8192, lcb_batch="test6 (175)",
                            lcb_max_private_tests=25)) as ex:
    arms = []
    for prec in ["q4_0", "f16"]:
        arms += [Arm(f"nothink_{prec}", prec), Arm(f"think_{prec}", prec, thinking=True)]
    res, comps = evaluate_arms(ex, arms, SUITES, reference="nothink", parallel=8, max_tokens=8192, ctx_per_slot=16384)
    ex.result(headline=" | ".join(f"{k}: lcb {v['lcb']['pass_rate']:.3f}" for k, v in res.items()))

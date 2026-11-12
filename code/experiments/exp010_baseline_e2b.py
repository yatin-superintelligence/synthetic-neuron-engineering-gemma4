"""EXP-010: unedited Gemma 4 E2B-it (QAT checkpoint) baselines at full precision (F16) and Q4_0.

Suites: MBPP+ dev half, MBPP+ test half, HumanEval+, HumanEvalFix (bug repair), RepoPatch v1 (EXP-020),
MMLU-500 (general control), behavioural probes v1 (40 everyday prompts).
Greedy decoding, no system prompt. Produces the failure taxonomy that decides which obstacles edits should target.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402

SUITES = ["mbpp_dev", "mbpp_test", "humaneval", "humanevalfix", "repopatch", "mmlu500", "behavior"]
PAR = int(os.environ.get("SNE_PARALLEL", "8"))

with Experiment("EXP-010", "baseline-e2b-qat-f16-and-q4", category="baseline", depends_on=["EXP-001", "EXP-002", "EXP-020"],
                motivation="Reference performance for every later comparison, and a failure taxonomy: training-free "
                           "edits are most likely to help where failures are mechanical (format, truncation, loops) "
                           "rather than missing knowledge (Takehi et al. 2026).",
                hypothesis="Q4_0 scores within a few points of F16. Most failures are wrong answers (logic), with a "
                           "minority of mechanical failures.",
                config=dict(suites=SUITES, decoding="greedy", parallel=PAR, system_prompt=None, max_tokens=8192, ctx_per_slot=16384, note="rerun without truncation; truncated first attempt archived")) as ex:
    arms = [Arm("base_f16", "f16"), Arm("base_q4_0", "q4_0")]
    res, comps = evaluate_arms(ex, arms, SUITES, reference="base_f16", parallel=PAR)
    h = []
    for a in ["base_f16", "base_q4_0"]:
        h.append(a + ": " + ", ".join(f"{s}={res[a][s]['pass_rate']:.3f}" for s in SUITES))
    ex.result(headline=" | ".join(h))

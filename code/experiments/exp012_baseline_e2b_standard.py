"""EXP-012: baselines for the standard (non-QAT) Gemma 4 E2B-it checkpoint, F16 and our own Q4_0 quantisation.

Why: generality. Edits are derived on the QAT checkpoint; replicating key results on a second checkpoint of the
same size guards against conclusions that hold for one set of weights only. It also measures how much QAT matters
under 4-bit quantisation (standard weights quantised with plain round-to-nearest Q4_0 vs Google's QAT Q4_0).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import MODELS, ROOT  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402

SUITES = ["mbpp_dev", "mbpp_test", "humaneval", "humanevalfix", "repopatch", "mmlu500", "behavior"]
PAR = int(os.environ.get("SNE_PARALLEL", "8"))
F16 = os.path.join(MODELS, "gguf", "e2b-std-f16.gguf")
Q4 = os.path.join(MODELS, "gguf", "e2b-std-q4_0.gguf")

with Experiment("EXP-012", "baseline-e2b-standard-f16-and-q4", category="baseline", depends_on=["EXP-010"],
                motivation="Generality across checkpoints, and the effect of QAT under Q4_0.",
                hypothesis="Standard F16 is within ~2 points of QAT F16; standard Q4_0 (no QAT) loses more than QAT Q4_0.",
                config=dict(suites=SUITES, f16=F16, q4=Q4, quantizer="llama-quantize Q4_0 (b12129)")) as ex:
    if not os.path.exists(Q4):
        cmd = [os.path.join(ROOT, "tools", "llama", "vulkan", "llama-quantize.exe"), F16, Q4, "Q4_0"]
        ex.log("CMD", " ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True)
        ex.log(p.stdout[-1500:], p.stderr[-1500:])
    arms = [Arm("std_f16", "f16", gguf=F16), Arm("std_q4_0", "q4_0", gguf=Q4)]
    res, comps = evaluate_arms(ex, arms, SUITES, reference="std_f16", parallel=PAR)
    ex.result(headline=" | ".join(a.name + ": " + ", ".join(f"{s}={res[a.name][s]['pass_rate']:.3f}" for s in SUITES)
                                  for a in arms))

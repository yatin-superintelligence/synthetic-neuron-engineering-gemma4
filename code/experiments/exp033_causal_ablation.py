"""EXP-033: causal test of the localisation. Zero-ablate code-selective neurons vs controls.

Arms per precision (F16 and Q4_0), all compiled as exact LoRA adapters (src/sne/lora.py: neuron_scale alpha=0):
  base                     unedited
  code_top{k}              top-k code-vs-prose neurons (Cohen's d, EXP-031), k in {256, 2048}
  rand{k}_s{0,1}           random neurons with the SAME per-layer counts as code_top{k}, excluding them
  prose_top{k}             top-k prose-vs-code neurons
Suites: MBPP+ dev (code), MMLU-500 (knowledge), behavioural probes (everyday behaviour).
A double dissociation (code arms hurt code more than MMLU/behaviour; prose arms the reverse; random arms neither)
is the evidence that the selected neurons are causally involved in coding rather than generic high-impact units.
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT  # noqa: E402
from sne.artifacts import build_adapters, publish  # noqa: E402
from sne.lora import neuron_scale  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402

KS = [256, 2048]
SUITES = ["mbpp_dev", "mmlu500", "behavior"]
PAR = int(os.environ.get("SNE_PARALLEL", "8"))
rank = json.load(open(glob.glob(os.path.join(ROOT, "experiments", "EXP-031_*", "neuron_rankings.json"))[0]))


def by_layer(items):
    d = {}
    for L, j in items:
        d.setdefault(L, []).append(j)
    return d


def ablate(sel):
    def fn(edit, bw, prec):
        for L, idx in sorted(by_layer(sel).items()):
            neuron_scale(edit, bw.get(L, "mlp.down_proj"), L, idx, 0.0)
        edit.meta.update(n_neurons=len(sel), layers=sorted({L for L, _ in sel}))
    return fn


with Experiment("EXP-033", "causal-ablation-code-neurons", category="localization", depends_on=["EXP-031", "EXP-010"],
                motivation="Selectivity is correlational. Ablation tests whether the code-selective neurons are "
                           "necessary for coding specifically, against layer-matched random neurons (controls for the "
                           "layer distribution) and prose-selective neurons (controls for 'any selective neuron').",
                hypothesis="Ablating top code neurons lowers MBPP+ dev pass rate far more than layer-matched random "
                           "ablation and more than prose-neuron ablation, while MMLU/behaviour degrade less.",
                config=dict(ks=KS, random_seeds=[0, 1], suites=SUITES, selection="EXP-031 cohen_d code vs prose",
                            alpha=0.0, parallel=PAR)) as ex:
    widths = rank["layer_widths"]
    arms_def = {}
    for k in KS:
        code = [(L, j) for L, j, _, _ in rank["code_top"][:k]]
        prose = [(L, j) for L, j, _, _ in rank["prose_top"][:k]]
        arms_def[f"code_top{k}"] = code
        arms_def[f"prose_top{k}"] = prose
        cnt = {L: len(v) for L, v in by_layer(code).items()}
        cset = set(code)
        for s in (0, 1):
            rng = random.Random(1000 * k + s)
            rnd = []
            for L, c in cnt.items():
                pool = [j for j in range(widths[L]) if (L, j) not in cset]
                rnd += [(L, j) for j in rng.sample(pool, c)]
            arms_def[f"rand{k}_s{s}"] = rnd
    ex.save_json("arm_neurons.json", {k: v for k, v in arms_def.items()})
    adapters = {name: build_adapters(ex, name, ablate(sel)) for name, sel in arms_def.items()}
    arms = []
    for prec in ["f16", "q4_0"]:
        arms.append(Arm(f"base_{prec}", prec))
        for name in arms_def:
            arms.append(Arm(f"{name}_{prec}", prec, lora=adapters[name][prec]))
    res, comps = evaluate_arms(ex, arms, SUITES, reference="base", parallel=PAR)
    rows = []
    for a in arms:
        rows.append(f"{a.name}: " + ", ".join(f"{s}={res[a.name][s]['pass_rate']:.3f}" for s in SUITES))
    ex.note("Pass rates per arm: " + " ; ".join(rows))
    publish(ex, [os.path.join(ROOT, "artifacts_local", ex.id)], f"{ex.id}_adapters.zip")
    ex.result(headline=" | ".join(r for r in rows if "f16" in r)[:600])

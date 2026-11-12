"""EXP-050: amplify code-selective MLP neurons (down_proj columns x alpha) and compare to matched random neurons.

Grid (pre-registered): k in {256, 1024} top code neurons (EXP-031 Cohen's d), alpha in {1.25, 1.5, 2.0}.
Control: for each (k, alpha), k layer-matched random neurons amplified by the same alpha (seed 0).
Suites: MBPP+ dev (selection metric) and MMLU-500 (side effects). Both precisions. Held-out evaluation of any
winner happens only in EXP-100.
Note on Gemma 4: the MLP output passes through post_feedforward_layernorm, so amplifying a subset re-weights it
relative to the other neurons; the RMS of the write is restored by the norm.
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

KS, ALPHAS = [256, 1024], [1.25, 1.5, 2.0]
SUITES = ["mbpp_dev", "mmlu500"]
PAR = int(os.environ.get("SNE_PARALLEL", "8"))
rank = json.load(open(glob.glob(os.path.join(ROOT, "experiments", "EXP-031_*", "neuron_rankings.json"))[0]))


def by_layer(items):
    d = {}
    for L, j in items:
        d.setdefault(L, []).append(j)
    return d


def scale(sel, alpha):
    def fn(edit, bw, prec):
        for L, idx in sorted(by_layer(sel).items()):
            neuron_scale(edit, bw.get(L, "mlp.down_proj"), L, idx, alpha)
        edit.meta.update(n_neurons=len(sel), alpha=alpha)
    return fn


with Experiment("EXP-050", "code-neuron-amplification-sweep", category="improvement",
                depends_on=["EXP-031", "EXP-033"],
                motivation="Simplest positive intervention: if code neurons carry coding computation, boosting their "
                           "relative contribution might sharpen it. Matched random amplification tells whether any "
                           "change is specific to the selected neurons or a generic perturbation effect.",
                hypothesis="Mild amplification (1.25) is near-neutral; strong amplification (2.0) degrades both code "
                           "and MMLU. Any dev gain over base is small (<3 points) and not larger than random.",
                config=dict(ks=KS, alphas=ALPHAS, control="layer-matched random, seed 0", suites=SUITES)) as ex:
    widths = rank["layer_widths"]
    defs = {}
    for k in KS:
        code = [(L, j) for L, j, _, _ in rank["code_top"][:k]]
        cset, cnt = set(code), {L: len(v) for L, v in by_layer(code).items()}
        rng = random.Random(5000 + k)
        rnd = [(L, j) for L, c in cnt.items() for j in rng.sample([x for x in range(widths[L]) if (L, x) not in cset], c)]
        for a in ALPHAS:
            defs[f"code{k}_a{a}"] = (code, a)
            defs[f"rand{k}_a{a}"] = (rnd, a)
    adapters = {n: build_adapters(ex, n, scale(sel, a)) for n, (sel, a) in defs.items()}
    arms = []
    for prec in ["f16", "q4_0"]:
        arms.append(Arm(f"base_{prec}", prec))
        arms += [Arm(f"{n}_{prec}", prec, lora=adapters[n][prec]) for n in defs]
    res, comps = evaluate_arms(ex, arms, SUITES, reference="base", parallel=PAR)
    table = {a.name: {s: res[a.name][s]["pass_rate"] for s in SUITES} for a in arms}
    best = max((n for n in defs if n.startswith("code")), key=lambda n: table[f"{n}_f16"]["mbpp_dev"])
    publish(ex, [os.path.join(ROOT, "artifacts_local", ex.id)], f"{ex.id}_adapters.zip")
    ex.result(table=table, best_code_arm_f16=best,
              headline=f"best code arm {best}: f16 dev {table[best + '_f16']['mbpp_dev']:.3f} vs base "
                       f"{table['base_f16']['mbpp_dev']:.3f}; q4 {table[best + '_q4_0']['mbpp_dev']:.3f} vs "
                       f"{table['base_q4_0']['mbpp_dev']:.3f}")

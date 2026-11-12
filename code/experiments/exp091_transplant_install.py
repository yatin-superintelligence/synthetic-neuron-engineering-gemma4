"""EXP-091 (numbered EXP-081 until 2026-11-08, never run): install donor code neurons (mapped by EXP-090/092) into E2B's least-used MLP slots and evaluate.

Arms per precision:
  base
  tx_top{n}_k{kappa}  : top-n donor code neurons per recipient layer (by donor d, filtered to read-fidelity > 0.3),
                        write scale kappa; n in {16, 64}, kappa in {0.5, 1.0} (n=16 only at kappa=1.0)
  tx_rand64_k1.0      : 64 random donor neurons per layer through the same maps (specificity control)
  recycle_only64      : the same 64 slots zeroed, nothing installed (cost of recycling)
Slots: lowest-usage neurons per layer from EXP-031 (low_usage_by_layer.json). Suites: MBPP+ dev, MMLU-500, behaviour.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT  # noqa: E402
from sne.artifacts import build_adapters, publish  # noqa: E402
from sne.lora import neuron_replace, neuron_scale  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402

DONOR = os.environ.get("SNE_DONOR", "e4b")
SRC = {"e4b": "EXP-090", "12b": "EXP-092"}[DONOR]
EXP_ID = {"e4b": "EXP-091", "12b": "EXP-093"}[DONOR]
SUITES = ["mbpp_dev", "mmlu500", "behavior"]
PAR = int(os.environ.get("SNE_PARALLEL", "8"))
mapped = torch.load(os.path.join(ROOT, "artifacts_local", SRC, "mapped_neurons.pt"))
low = json.load(open(glob.glob(os.path.join(ROOT, "experiments", "EXP-031_*", "low_usage_by_layer.json"))[0]))


def pick(L, n, fid_min=0.3):
    m = mapped[L]
    ok = [i for i, f in enumerate(m["fidelity"]) if f > fid_min][:n]
    return ok


def install(n, kappa, which="top"):
    def fn(edit, bw, prec):
        info = {}
        for L, m in mapped.items():
            sel = pick(L, n) if which == "top" else list(range(min(n, len(m["rand_idx"]))))
            Wg, Wu, Wd = bw.get(L, "mlp.gate_proj"), bw.get(L, "mlp.up_proj"), bw.get(L, "mlp.down_proj")
            slots = low[str(L)][:len(sel)]
            for s, i in zip(slots, sel):
                g = m["g"][i] if which == "top" else m["rand_g"][i]
                u = m["u"][i] if which == "top" else m["rand_u"][i]
                d = (m["d"][:, i] if which == "top" else m["rand_d"][:, i]) * kappa
                neuron_replace(edit, Wg, Wu, Wd, L, s, g, u, d)
            info[L] = dict(n=len(sel), slots=slots, donor_layer=m["donor_layer"])
        edit.meta.update(donor=DONOR, which=which, n=n, kappa=kappa, layers=info)
    return fn


def recycle(n):
    def fn(edit, bw, prec):
        for L in mapped:
            neuron_scale(edit, bw.get(L, "mlp.down_proj"), L, low[str(L)][:n], 0.0)
        edit.meta.update(recycle_only=n)
    return fn


with Experiment(EXP_ID, f"transplant-install-{DONOR}-to-e2b", category="transplant", depends_on=[SRC, "EXP-031"],
                motivation="Tests whether neurons transplanted from a stronger sibling add coding ability to E2B, with "
                           "controls for random donor neurons (any added capacity) and for the cost of freeing slots.",
                hypothesis="Recycling the least-used slots is near-free. Transplanted code neurons change dev pass rate "
                           "by at most a few points; any gain must exceed the random-donor control to count.",
                config=dict(donor=DONOR, source=SRC, n=[16, 64], kappa=[0.5, 1.0], fidelity_min=0.3, suites=SUITES)) as ex:
    defs = {"tx_top64_k1.0": install(64, 1.0), "tx_top64_k0.5": install(64, 0.5), "tx_top16_k1.0": install(16, 1.0),
            "tx_rand64_k1.0": install(64, 1.0, "rand"), "recycle_only64": recycle(64)}
    adapters = {n: build_adapters(ex, n, f) for n, f in defs.items()}
    arms = []
    for prec in ["f16", "q4_0"]:
        arms.append(Arm(f"base_{prec}", prec))
        arms += [Arm(f"{n}_{prec}", prec, lora=adapters[n][prec]) for n in defs]
    res, comps = evaluate_arms(ex, arms, SUITES, reference="base", parallel=PAR)
    table = {a.name: {s: res[a.name][s]["pass_rate"] for s in SUITES} for a in arms}
    publish(ex, [os.path.join(ROOT, "artifacts_local", ex.id)], f"{ex.id}_adapters.zip")
    ex.result(table=table, headline="; ".join(f"{n}: dev {v['mbpp_dev']:.3f} mmlu {v['mmlu500']:.3f}"
                                              for n, v in table.items() if "f16" in n))

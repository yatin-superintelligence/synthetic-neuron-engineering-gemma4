"""EXP-056: synthetic gated neurons (closed form, no training) installed in recycled MLP slots.

Each synthetic neuron fires on code-context tokens (gate = linear discriminant code vs prose, with emulated bias)
and writes a target direction into the residual stream (through the post-FFN RMSNorm, compensated analytically).
Target direction v_L = mean residual on PASSING code minus FAILING code at layer L (EXP-031, pass/fail corpora).
Arms (per precision): gated rho in {0.05, 0.1, 0.2}; always-on rho=0.1; gated random-direction rho=0.1 (control).
Layers: the 4 layers in 6..28 with the largest ||mu_pass - mu_fail|| / ||mu_code||, one neuron each (slot = the
lowest-usage neuron of that layer, EXP-031). Calibration uses task half A only; fidelity is measured on half B.
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, ROOT  # noqa: E402
from sne.artifacts import build_adapters, publish  # noqa: E402
from sne.lora import neuron_replace  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402
from sne.synth import build_neuron, capture, fit_const_dir, fit_gate, gelu_tanh  # noqa: E402

E31 = glob.glob(os.path.join(ROOT, "experiments", "EXP-031_*"))[0]
CORPUS = json.load(open(glob.glob(os.path.join(ROOT, "experiments", "EXP-030_*", "corpus.json"))[0], encoding="utf-8"))
SUITES = ["mbpp_dev", "mmlu500", "behavior"]
PAR = int(os.environ.get("SNE_PARALLEL", "8"))
RHOS = [0.05, 0.1, 0.2]

with Experiment("EXP-056", "synthetic-gated-neurons-pass-direction", category="improvement",
                depends_on=["EXP-031", "EXP-033"],
                motivation="Tests whether new neurons, written in closed form, can add a conditional behaviour: in code "
                           "context, push the residual stream toward the mean state of the model's own *correct* code. "
                           "Controls separate the effect of the gate (always-on) and of the direction (random).",
                hypothesis="Gate fidelity on held-out tokens is high (AUC > 0.9). Small rho is neutral; some rho gives a "
                           "small dev gain over base and over the random-direction control; always-on hurts MMLU more "
                           "than gated.",
                config=dict(rhos=RHOS, beta=4.0, gamma=1.0, n_layers=4, layer_range=[6, 28], calib_seqs_per_class=60,
                            target="mu_pass - mu_fail residual after layer L", suites=SUITES)) as ex:
    resid = np.load(os.path.join(E31, "resid_means.npy"))  # [corpus(code_A,code_B,prose_A,prose_B,pass,fail), L, D]
    low = json.load(open(os.path.join(E31, "low_usage_neurons.json")))
    code_mu = (resid[0] + resid[1]) / 2
    score = np.linalg.norm(resid[4] - resid[5], axis=1) / np.linalg.norm(code_mu, axis=1)
    cand = [L for L in range(6, 29)]
    layers_sel = sorted(sorted(cand, key=lambda L: -score[L])[:4])
    slots = {}
    for L, j, u in low:
        if L in layers_sel and L not in slots:
            slots[L] = j
    ex.log("layers", layers_sel, "slots", slots, "score", [round(float(score[L]), 4) for L in layers_sel])
    # calibration (half A) and held-out (half B) token sets
    idx = list(range(len(CORPUS)))
    random.Random(56).shuffle(idx)
    A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
    pick = lambda ids, key, prompt: [(CORPUS[i][prompt], CORPUS[i][key]) for i in ids[:60]]  # noqa: E731
    from sne.model import load
    model, tok = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"), ple_on_cpu=True)
    cal_c = capture(model, tok, pick(A, "code_greedy", "code_prompt"), layers_sel)
    cal_p = capture(model, tok, pick(A, "prose", "prose_prompt"), layers_sel)
    hold_c = capture(model, tok, pick(B, "code_greedy", "code_prompt"), layers_sel, want_out=False)
    hold_p = capture(model, tok, pick(B, "prose", "prose_prompt"), layers_sel, want_out=False)
    Ls = model.model.language_model.layers
    neurons, fidelity = {}, {}
    rng = torch.Generator().manual_seed(56)
    for L in layers_sel:
        Xc, Xp = cal_c[L]["x"], cal_p[L]["x"]
        z, zq = fit_const_dir(torch.cat([Xc, Xp]))
        w, t, gq = fit_gate(Xc, Xp)
        rms_y = float(cal_c[L]["y"].pow(2).mean().sqrt())
        w_post = Ls[L].post_feedforward_layernorm.weight.detach().float().cpu()
        v = torch.tensor(resid[4][L] - resid[5][L], dtype=torch.float32)
        v_rand = torch.randn(v.shape, generator=rng)
        for rho in RHOS:
            neurons[("gated", rho, L)] = build_neuron(z, w_post, rms_y, v, rho, gate=(w, t), X_on=Xc)
        neurons[("ungated", 0.1, L)] = build_neuron(z, w_post, rms_y, v, 0.1, gate=None, X_on=Xc)
        neurons[("gated_rand", 0.1, L)] = build_neuron(z, w_post, rms_y, v_rand, 0.1, gate=(w, t), X_on=Xc)
        g, u, d, info = neurons[("gated", 0.1, L)]
        ac = gelu_tanh(hold_c[L]["x"] @ g) * (hold_c[L]["x"] @ u)
        ap = gelu_tanh(hold_p[L]["x"] @ g) * (hold_p[L]["x"] @ u)
        _, _, gq_hold = fit_gate(hold_c[L]["x"], hold_p[L]["x"])
        thr = 0.5 * float(ac.mean())
        fidelity[L] = dict(const_dir=zq, gate_calib=gq, heldout_mean_act_code=float(ac.mean()),
                           heldout_mean_act_prose=float(ap.mean()), heldout_frac_on_code=float((ac > thr).float().mean()),
                           heldout_frac_on_prose=float((ap > thr).float().mean()), slot=slots[L], **info)
        ex.log("layer", L, fidelity[L])
    ex.save_json("fidelity.json", fidelity)
    torch.save({f"{k[0]}|{k[1]}|{k[2]}": v[:3] for k, v in neurons.items()}, os.path.join(ex.dir, "neurons.pt"))
    del model
    import gc
    gc.collect()
    torch.xpu.empty_cache()
    variants = [("gated", r) for r in RHOS] + [("ungated", 0.1), ("gated_rand", 0.1)]

    def make(kind, rho):
        def fn(edit, bw, prec):
            for L in layers_sel:
                g, u, d, _ = neurons[(kind, rho, L)]
                neuron_replace(edit, bw.get(L, "mlp.gate_proj"), bw.get(L, "mlp.up_proj"), bw.get(L, "mlp.down_proj"),
                               L, slots[L], g, u, d)
            edit.meta.update(kind=kind, rho=rho, layers=layers_sel, slots=slots)
        return fn
    adapters = {f"{k}_r{r}": build_adapters(ex, f"{k}_r{r}", make(k, r)) for k, r in variants}
    arms = []
    for prec in ["f16", "q4_0"]:
        arms.append(Arm(f"base_{prec}", prec))
        arms += [Arm(f"{n}_{prec}", prec, lora=adapters[n][prec]) for n in adapters]
    res, comps = evaluate_arms(ex, arms, SUITES, reference="base", parallel=PAR)
    table = {a.name: {s: res[a.name][s]["pass_rate"] for s in SUITES} for a in arms}
    publish(ex, [os.path.join(ROOT, "artifacts_local", ex.id)], f"{ex.id}_adapters.zip")
    ex.result(table=table, layers=layers_sel, headline="; ".join(f"{n}: dev {v['mbpp_dev']:.3f} mmlu {v['mmlu500']:.3f}"
                                                                 for n, v in table.items() if "f16" in n))

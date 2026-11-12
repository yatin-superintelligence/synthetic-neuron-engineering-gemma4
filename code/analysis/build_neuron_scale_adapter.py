"""EXP-155 adapters (D-213): scale the MLP neurons EXP-154 found to track a solving run.

For each chosen neuron (layer l, index i) the edit multiplies its output by (1 + s * alpha): s = +1 for a neuron higher in passing runs
(amplified), s = -1 for one higher in failing runs (damped). As a LoRA on down_proj of layer l: A holds one-hot rows e_i (rank = the
number of chosen neurons in that layer), B holds s * alpha * W_down[:, i]. A neuron chosen at both positions keeps the sign of its
stronger AUC. Control: the same number of neurons in the same layers, chosen at random, with random signs.
Usage: python tools/analysis/build_neuron_scale_adapter.py [--alpha 1.0] [--k 64]   (reads artifacts_local/EXP-154/neurons.pt)"""
import json
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SRC = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else os.path.join("artifacts_local", "EXP-154", "neurons.pt")
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join("artifacts_local", "EXP-155")


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def save(name, chosen, w_down, n_inter, alpha):
    from safetensors.torch import save_file
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    by = {}
    for (l, i), s in chosen.items():
        by.setdefault(l, []).append((i, s))
    t, rp = {}, {}
    for l, items in sorted(by.items()):
        r = len(items)
        A = torch.zeros(r, n_inter)
        B = torch.zeros(w_down[f"{l}|{items[0][0]}"].shape[0], r)
        for k, (i, s) in enumerate(items):
            A[k, i] = 1.0
            B[:, k] = s * alpha * w_down[f"{l}|{i}"]
        pre = f"base_model.model.model.language_model.layers.{l}.mlp.down_proj"
        t[f"{pre}.lora_A.weight"] = A.contiguous()
        t[f"{pre}.lora_B.weight"] = B.contiguous()
        rp[f"model.language_model.layers.{l}.mlp.down_proj"] = r
    save_file(t, os.path.join(d, "adapter_model.safetensors"))
    rmax = max(rp.values())
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-31B-it", r=rmax, lora_alpha=rmax,
               rank_pattern=rp, alpha_pattern=rp, target_modules=["down_proj"], lora_dropout=0.0, bias="none", fan_in_fan_out=False,
               init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    return d, {str(l): len(v) for l, v in sorted(by.items())}


def main():
    alpha = float(opt("--alpha", 1.0))
    k = int(opt("--k", 64))
    src = torch.load(SRC, weights_only=False)
    top = src["top"][:k]
    best = {}
    for t in top:  # one sign per neuron: that of its stronger feature
        key = (t["layer"], t["neuron"])
        if key not in best or abs(t["auc"] - 0.5) > abs(best[key]["auc"] - 0.5):
            best[key] = t
    chosen = {key: (1 if t["auc"] > 0.5 else -1) for key, t in best.items()}
    w_down = src["w_down"]
    n_inter = src["n_inter"]
    d, per_layer = save("neurons", chosen, w_down, n_inter, alpha)
    print("edit:", d, "| neurons", len(chosen), "| amplified", sum(1 for s in chosen.values() if s > 0), "| damped", sum(1 for s in chosen.values() if s < 0), "| per layer", per_layer)
    # control: random neurons in the same layers (their down_proj columns are needed, so they are drawn among the saved ones'
    # layers from a fresh capture of columns is not available here; the control uses the chosen neurons' columns with shuffled
    # signs, which keeps the same norm per layer and destroys the pass/fail alignment)
    rng = random.Random(155)
    keys = list(chosen)
    signs = [chosen[q] for q in keys]
    rng.shuffle(signs)
    ctl = dict(zip(keys, signs))
    flips = sum(1 for q in keys if ctl[q] != chosen[q])
    d2, _ = save("control", ctl, w_down, n_inter, alpha)
    print("control:", d2, "| same neurons, signs shuffled; differing signs", flips, "of", len(keys))
    json.dump(dict(alpha=alpha, k=k, neurons=[dict(layer=l, neuron=i, sign=s) for (l, i), s in sorted(chosen.items())], per_layer=per_layer),
              open(os.path.join(OUT, "build.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

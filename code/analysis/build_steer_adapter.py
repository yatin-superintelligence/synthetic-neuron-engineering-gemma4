"""EXP-151 adapters (D-202): an always-on unit that pushes 31B's state along EXP-147's success direction, built on the laptop from
EXP-147's capture (artifacts_local/EXP-147/capture.pt, fit.json); no GPU.

The unit replaces the least-used neuron i of the MLP at the chosen depth L:
  gate row K b, up row b, where b . x is about 1 at every position (ridge fit on the saved MLP inputs), so the unit's activation is
  about GELU(K) at every position;
  down column u = beta (d / w_post), d the unit success direction in the layer's output (task-centred mean difference, all rows).
Gemma 4 normalises each MLP's output (post_feedforward_layernorm) before adding it to the residual stream, so a write larger than the
MLP's own output replaces it. The change to the layer output is therefore computed exactly on the saved MLP outputs y:
  delta = layer_scalar * (norm(y - a_old w_down + a_new u) - norm(y)), norm(v) = v / rms(v) * w_post,
and beta is set by bisection so that the mean of delta . d equals c times the solved-failed gap along d. Reported per strength: the
achieved shift, the part of delta off the direction, and |a_new u| / |y| (above 1 means the layer's MLP output is being replaced).
Control: a random direction orthogonal to d, calibrated to the same shift along itself.
Usage: python tools/analysis/build_steer_adapter.py [--layer L] [--kind last|outmean] [--c 1,2]"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
CAP = sys.argv[sys.argv.index("--cap") + 1] if "--cap" in sys.argv else os.path.join("artifacts_local", "EXP-147", "capture.pt")
FIT = sys.argv[sys.argv.index("--fit") + 1] if "--fit" in sys.argv else os.path.join("artifacts_local", "EXP-147", "fit.json")
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join("artifacts_local", "EXP-151")
K = 3.0
R_MAX = 1.0


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def save_lora(name, layer, i, dgate, dup, ddown, n_inter):
    from safetensors.torch import save_file
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    pre = f"base_model.model.model.language_model.layers.{layer}.mlp"
    e = torch.zeros(n_inter, 1)
    e[i, 0] = 1.0
    t = {f"{pre}.gate_proj.lora_A.weight": dgate.view(1, -1).float().contiguous(), f"{pre}.gate_proj.lora_B.weight": e.clone(),
         f"{pre}.up_proj.lora_A.weight": dup.view(1, -1).float().contiguous(), f"{pre}.up_proj.lora_B.weight": e.clone(),
         f"{pre}.down_proj.lora_A.weight": e.view(1, -1).clone(), f"{pre}.down_proj.lora_B.weight": ddown.view(-1, 1).float().contiguous()}
    save_file(t, os.path.join(d, "adapter_model.safetensors"))
    rp = {f"model.language_model.layers.{layer}.mlp.{m}": 1 for m in ("gate_proj", "up_proj", "down_proj")}
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-31B-it", r=1, lora_alpha=1, rank_pattern=rp,
               alpha_pattern={k: 1 for k in rp}, target_modules=["gate_proj", "up_proj", "down_proj"], lora_dropout=0.0, bias="none",
               fan_in_fan_out=False, init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    return d


def gelu(z):
    return 0.5 * z * (1 + torch.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def main():
    cap = torch.load(CAP, weights_only=False)
    fit = json.load(open(FIT, encoding="utf-8"))
    L = int(opt("--layer", fit["best"]["layer"]))
    kind = opt("--kind", fit["best"]["position"])
    cs = [float(c) for c in opt("--c", "1,2").split(",")]
    rows = cap["rows"]
    solved = np.array([r["solved"] for r in rows], bool)
    task = np.array([r["task"] for r in rows])
    X = cap["feats"][f"{L}|{kind}"].double().numpy()
    Xc = X.copy()
    for t in set(task.tolist()):
        m = task == t
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    dvec = Xc[solved].mean(0) - Xc[~solved].mean(0)
    d = torch.tensor(dvec / np.linalg.norm(dvec))
    gap = float((Xc[solved] @ d.numpy()).mean() - (Xc[~solved] @ d.numpy()).mean())
    u0 = cap["units"][L]
    xs = cap["samples"][f"{L}|x"].double()
    ys = cap["samples"][f"{L}|y"].double()
    lam = 1e-2 * float((xs ** 2).sum(1).mean())
    b = torch.linalg.solve(xs.T @ xs + lam * torch.eye(xs.shape[1], dtype=torch.float64), xs.T @ torch.ones(len(xs), dtype=torch.float64))
    bx = xs @ b
    wpost, eps, ls = u0["w_post"].double(), u0["eps"], u0["layer_scalar"]
    wg, wu, wd = u0["w_gate"].double(), u0["w_up"].double(), u0["w_down"].double()
    a_old = gelu(xs @ wg) * (xs @ wu)
    a_new = gelu(K * bx) * bx

    def norm(v):
        return v / torch.sqrt((v ** 2).mean(-1, keepdim=True) + eps) * wpost

    base = norm(ys)
    ynorm = ys.norm(dim=1)

    def effect(u, direction):
        y2 = ys - a_old[:, None] * wd[None] + a_new[:, None] * u[None]
        delta = ls * (norm(y2) - base)
        along = delta @ direction
        off = (delta - along[:, None] * direction[None]).norm(dim=1)
        return float(along.mean()), float(off.mean()), float(((a_new[:, None] * u[None]).norm(dim=1) / ynorm).median())

    floor = 0.05 * float(wpost.detach().abs().median())
    wsafe = torch.where(wpost.abs() < floor, torch.sign(wpost + 1e-12) * floor, wpost)
    report = dict(layer=L, kind=kind, unit=u0["i"], gap=round(gap, 4), b_fit=dict(mean=round(float(bx.mean()), 4), sd=round(float(bx.std()), 4)),
                  a_new=dict(mean=round(float(a_new.mean()), 4), sd=round(float(a_new.std()), 4)), samples=len(xs), adapters={})
    g = torch.Generator().manual_seed(151)
    r = torch.randn(d.shape, generator=g, dtype=torch.float64)
    r = r - (r @ d) * d
    r = r / r.norm()
    for name, direction, c in [(f"steer_c{c:g}", d, c) for c in cs] + [(f"random_c{max(cs):g}", r, max(cs))]:
        target = c * gap
        dirw = direction / wsafe
        # the write may not exceed the MLP's own output (median |a_new u| / |y| at most R_MAX): past that it replaces the layer's MLP
        # output instead of adding to it; a target that needs more is capped there and reported as capped
        beta_max = R_MAX / max(effect(dirw, direction)[2], 1e-12)
        lo, hi = 0.0, min(1.0, beta_max)
        while effect(hi * dirw, direction)[0] < target and hi < beta_max:
            hi = min(hi * 2, beta_max)
        for _ in range(60):
            mid = (lo + hi) / 2
            if effect(mid * dirw, direction)[0] < target:
                lo = mid
            else:
                hi = mid
        u = hi * dirw
        along, off, ratio = effect(u, direction)
        save_lora(name, L, u0["i"], (K * b - wg).float(), (b - wu).float(), (u - wd).float(), u0["n_inter"])
        report["adapters"][name] = dict(target_shift=round(target, 4), shift=round(along, 4), off_direction=round(off, 4),
                                        write_over_mlp_output_median=round(ratio, 3), beta=round(hi, 6), capped=bool(along < 0.98 * target))
        print(name, report["adapters"][name])
    json.dump(report, open(os.path.join(OUT, "build.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "adapters"}))


if __name__ == "__main__":
    main()

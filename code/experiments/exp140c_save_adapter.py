"""EXP-140c's neuron saved as an adapter for the agent run EXP-140a (D-176), from the same cached activations and weights, without a
model pass. The read-out is refitted exactly as exp140c_31b_delimiter_competitors.py fits it (the same rows, fold rule, sample of
markdown backticks, ridge strength and design); the write is alpha * g / |g| with EXP-140c's competitor set and grid strength.
Run on the runtime that holds /content/exp140_acts.pt and /content/hf_31b. Prints the read-out at the faulty states and the norms so
they can be checked against EXP-140c's log. Writes /content/exp140c_adapter/{delim,delim_control}."""
import json
import os
import zlib

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CACHE, D, OUT = "/content/exp140_acts.pt", "/content/hf_31b", "/content/exp140c_adapter"
Q, BTC, TICK = 52, 8347, 236929
COMP = [8347, 236929, 148, 228276]  # EXP-140c round 2
GRID = np.geomspace(0.1, 20.0, 41)
ALPHA_LOGGED = 4.0806
LAM, TXT_CAP, LAYER = 0.01, 3000, 59
NEG = ("CONT", "CONT_H", "CONT_BT", "TXT_BT", "TXT_BTC", "OTHER")
FOLD_A = {"fastapi_9753", "rich_3470", "rich_3454", "fastapi_11194", "fastapi_14794"}
FOLD_B = {"rich_3944", "fastapi_14258", "rich_3067", "rich_3472"}


def fold(iid):
    return 0 if iid in FOLD_A else 1 if iid in FOLD_B else zlib.crc32(iid.encode()) % 2


c0 = torch.load(CACHE, weights_only=False)
X, cls_all = c0["X"], np.array(c0["cls"])
rng = np.random.default_rng(140)
fo_all = np.array([fold(t) for t in c0["iid"]])
use = np.ones(len(cls_all), bool)
tb = np.where(cls_all == "TXT_BT")[0]
if len(tb) > TXT_CAP:
    use[rng.choice(tb, len(tb) - TXT_CAP, replace=False)] = False
sel = use & np.isin(cls_all, ("ERR",) + NEG)
idx = np.where(sel)[0]
t = torch.tensor((cls_all[idx] == "ERR").astype(np.float64))
w = torch.ones(len(idx), dtype=torch.float64)
m = t == 1
w[m] = float((~m).sum()) / float(m.sum())
Xs = X[idx].to("cuda", torch.float64) * w.to("cuda").sqrt()[:, None]
K = Xs @ Xs.T
a_ = torch.linalg.solve(K + LAM * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype, device=K.device), w.to("cuda").sqrt() * t.to("cuda"))
v = (Xs.T @ a_).float().cpu()
del Xs, K
err = cls_all == "ERR"
print("read-out at the 62 faulty states, in sample: mean", round(float((X[torch.as_tensor(err)].float() @ v).mean()), 4), "| v norm", round(float(v.norm()), 4))

idx_map = json.load(open(os.path.join(D, "model.safetensors.index.json")))["weight_map"]


def get(k, rows=None):
    with safe_open(os.path.join(D, idx_map[k]), "pt") as f:
        if rows is None:
            return f.get_tensor(k).float()
        s = f.get_slice(k)
        return torch.stack([s[r] for r in rows]).float()


pre = "model.language_model."
wpf = get(pre + f"layers.{LAYER}.post_feedforward_layernorm.weight")
wfin = get(pre + "norm.weight")
E = get(pre + "embed_tokens.weight", rows=[Q] + COMP)
e = E[0] - E[1:].mean(0)
g = e * wpf * wfin
alpha = float(GRID[np.argmin(np.abs(GRID - ALPHA_LOGGED))])
u = g / g.norm() * alpha
torch.manual_seed(140)
r = torch.randn_like(v)
r = r / r.norm() * v.norm()
print("alpha", alpha, "(logged", ALPHA_LOGGED, ") | u norm", round(float(u.norm()), 4), "| control read-out at faulty states", round(float((X[torch.as_tensor(err)].float() @ r).mean()), 4))


def save(name, vv):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    key = f"base_model.model.model.language_model.layers.{LAYER}.mlp.down_proj"
    save_file({f"{key}.lora_A.weight": vv.view(1, -1).float().contiguous(), f"{key}.lora_B.weight": u.view(-1, 1).float().contiguous()},
              os.path.join(d, "adapter_model.safetensors"))
    rp = {f"model.language_model.layers.{LAYER}.mlp.down_proj": 1}
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-31B-it", r=1, lora_alpha=1, rank_pattern=rp,
                   alpha_pattern={k: 1 for k in rp}, target_modules=["down_proj"], lora_dropout=0.0, bias="none", fan_in_fan_out=False,
                   init_lora_weights=True, inference_mode=True, use_rslora=False), open(os.path.join(d, "adapter_config.json"), "w"), indent=2)
    print("saved", d, sorted(os.listdir(d)))


save("delim", v)
save("delim_control", r)

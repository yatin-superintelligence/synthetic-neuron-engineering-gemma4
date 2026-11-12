"""Activation statistics in PyTorch (HF reference implementation, bf16 on XPU).

For every decoder layer we record, over response tokens only:
  * MLP neuron activations a = act(gate(x)) * up(x)  (the input of down_proj): sum, sum of squares, sum |a|,
    count(a > 0), count of tokens.
  * residual stream after the layer (sum, for mean directions) and the MLP input x (normed, for gate directions).
"""
import numpy as np
import torch

from .model import device as _device, empty_cache as _empty_cache

from .model import layers


class StatHooks:
    def __init__(self, model, want_resid=True):
        self.L = layers(model)
        self.mask = None
        self.stats = {}
        self.handles = []
        self.want_resid = want_resid
        for i, layer in enumerate(self.L):
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._mk_act(i)))
            if want_resid:
                self.handles.append(layer.mlp.register_forward_pre_hook(self._mk_in(i)))
                self.handles.append(layer.register_forward_hook(self._mk_out(i)))

    def _acc(self, key, val):
        if key in self.stats:
            self.stats[key] += val
        else:
            self.stats[key] = val.clone()

    def _sel(self, x):
        m = self.mask  # (B, T) bool
        return x[m].float()  # (N, D)

    def _mk_act(self, i):
        def hook(mod, args):
            a = self._sel(args[0])
            self._acc(f"act_sum_{i}", a.sum(0))
            self._acc(f"act_sq_{i}", (a * a).sum(0))
            self._acc(f"act_abs_{i}", a.abs().sum(0))
            self._acc(f"act_pos_{i}", (a > 0).float().sum(0))
            self._acc(f"n_{i}", torch.tensor(float(a.shape[0]), device=a.device))
        return hook

    def _mk_in(self, i):
        def hook(mod, args):
            self._acc(f"mlpin_sum_{i}", self._sel(args[0]).sum(0))
        return hook

    def _mk_out(self, i):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            self._acc(f"resid_sum_{i}", self._sel(h).sum(0))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def numpy(self):
        return {k: v.detach().cpu().numpy() for k, v in self.stats.items()}


def collect(model, tok, pairs, batch_size=4, max_len=1024, want_resid=True, dev=None):
    """pairs: list of (rendered_prompt_without_bos, response_text). Statistics over response tokens only."""
    dev = dev or _device()
    from . import SMOKE
    if SMOKE:
        pairs = pairs[:4]
    hooks = StatHooks(model, want_resid)
    try:
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            ids, masks = [], []
            for p, r in chunk:
                a = tok("<bos>" + p, add_special_tokens=False)["input_ids"]
                b = tok(r + "<turn|>", add_special_tokens=False)["input_ids"]
                seq = (a + b)[:max_len]
                ids.append(seq)
                masks.append([0] * len(a) + [1] * (len(seq) - len(a)))
            T = max(len(s) for s in ids)
            pad = tok.pad_token_id or 0
            inp = torch.tensor([s + [pad] * (T - len(s)) for s in ids], device=dev)
            att = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in ids], device=dev)
            hooks.mask = torch.tensor([m + [0] * (T - len(m)) for m in masks], device=dev).bool()
            with torch.no_grad():
                model(input_ids=inp, attention_mask=att, use_cache=False)
    finally:
        hooks.remove()
    return hooks.numpy()


def neuron_table(stats, n_layers):
    """Per-layer dict of mean, std, mean|a|, p(a>0)."""
    out = []
    for i in range(n_layers):
        n = float(stats[f"n_{i}"])
        mu = stats[f"act_sum_{i}"] / n
        var = np.maximum(stats[f"act_sq_{i}"] / n - mu ** 2, 1e-12)
        out.append(dict(n=n, mean=mu, std=np.sqrt(var), mean_abs=stats[f"act_abs_{i}"] / n,
                        p_pos=stats[f"act_pos_{i}"] / n, rms=np.sqrt(stats[f"act_sq_{i}"] / n)))
    return out


def selectivity(tab_a, tab_b, metric="cohen_d"):
    """Per-layer selectivity of each neuron for corpus A over corpus B."""
    res = []
    for a, b in zip(tab_a, tab_b):
        if metric == "cohen_d":
            pooled = np.sqrt(0.5 * (a["std"] ** 2 + b["std"] ** 2)) + 1e-6
            res.append((a["mean"] - b["mean"]) / pooled)
        elif metric == "rms_log_ratio":
            res.append(np.log((a["rms"] + 1e-6) / (b["rms"] + 1e-6)))
        elif metric == "ppos_diff":
            res.append(a["p_pos"] - b["p_pos"])
        else:
            raise ValueError(metric)
    return res

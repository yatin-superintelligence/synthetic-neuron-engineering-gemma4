"""Synthetic neurons: closed-form construction of new MLP neurons, installed into recycled slots.

A Gemma 4 MLP neuron computes  a = gelu_tanh(g . x) * (u . x)  and writes  a * d  into the MLP output y, which is
then RMS-normalised (post_feedforward_layernorm, weight w_post) before the residual add. There are no biases.

Construction of a *gated* neuron that fires on a context class C (e.g. code) and writes a target direction v:
  1. Bias emulation. Fit z with  z . x ~= 1  over calibration tokens (ridge). Then  b * (z . x) ~= b  acts as a
     bias. Its quality (R^2) is reported; MLP inputs are RMS-normalised, so a near-constant direction usually exists.
  2. Gate. Linear discriminant w = (Sigma + lam I)^-1 (mu_C - mu_notC); threshold t at the midpoint of the projected
     class means. gate row g = beta * (w - t z), so g . x ~= beta * (w . x - t).
  3. Up. u = gamma * z, so u . x ~= gamma (a constant amplitude). Hence a ~= gamma * gelu(beta * s), s = w.x - t.
  4. Down. To shift the post-norm residual by delta (small), the pre-norm MLP output must move by
     dy ~= rms(y) * delta / w_post (elementwise). With E[a | C] = abar:  d = rms(y) * delta / (w_post * abar).
  5. delta = rho * ||w_post|| / sqrt(D) * sqrt(D) * v_hat  -> strength rho is relative to the typical size of the
     MLP's own normalised contribution, so it is comparable across layers.
An *ungated* (always-on) neuron uses g = beta * z instead (s ~= beta constant), i.e. baked unconditional steering.
"""
import numpy as np
import torch

from .model import device as _device, empty_cache as _empty_cache

from .model import layers


@torch.no_grad()
def capture(model, tok, pairs, layer_ids, max_tokens_per_seq=256, dev=None, want_out=True):
    dev = dev or _device()
    """MLP inputs x (after pre_feedforward_layernorm) and MLP raw outputs y at layer_ids, response tokens only.
    Returns {L: {"x": [N, D] float32 cpu, "y": [N, D]}} with up to max_tokens_per_seq tokens per sequence."""
    L = layers(model)
    bufs = {i: {"x": [], "y": []} for i in layer_ids}
    state = {}

    def mk(i):
        def hook(mod, args, out):
            m = state["mask"]
            bufs[i]["x"].append(args[0][m].float().cpu())
            if want_out:
                bufs[i]["y"].append(out[m].float().cpu())
        return hook
    hs = [L[i].mlp.register_forward_hook(mk(i)) for i in layer_ids]
    try:
        for p, r in pairs:
            a = tok("<bos>" + p, add_special_tokens=False)["input_ids"]
            b = tok(r + "<turn|>", add_special_tokens=False)["input_ids"][:max_tokens_per_seq]
            ids = torch.tensor([a + b], device=dev)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            mask[0, len(a):] = True
            state["mask"] = mask
            model(input_ids=ids, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return {i: {k: torch.cat(v) if v else None for k, v in bufs[i].items()} for i in layer_ids}


def fit_const_dir(X, lam=1e-2):
    """z minimising ||X z - 1||^2 + lam ||z||^2. Returns z and R^2-like quality (1 - var(resid)/1)."""
    D = X.shape[1]
    A = X.T @ X + lam * X.shape[0] * torch.eye(D) / D
    z = torch.linalg.solve(A, X.T @ torch.ones(X.shape[0]))
    pred = X @ z
    return z, dict(mean=float(pred.mean()), std=float(pred.std()))


def fit_gate(Xc, Xn, lam=0.1):
    mu_c, mu_n = Xc.mean(0), Xn.mean(0)
    Xall = torch.cat([Xc - mu_c, Xn - mu_n])
    S = Xall.T @ Xall / Xall.shape[0]
    S = S + lam * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0])
    w = torch.linalg.solve(S, mu_c - mu_n)
    sc, sn = Xc @ w, Xn @ w
    scale = 1.0 / float(torch.cat([sc, sn]).std())
    w, sc, sn = w * scale, sc * scale, sn * scale
    t = float((sc.mean() + sn.mean()) / 2)
    # AUC of the projected score
    allv = torch.cat([sc, sn]).numpy()
    lab = np.r_[np.ones(len(sc)), np.zeros(len(sn))]
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    auc = (ranks[lab == 1].sum() - len(sc) * (len(sc) + 1) / 2) / (len(sc) * len(sn))
    return w, t, dict(auc=float(auc), sep=float(sc.mean() - sn.mean()))


def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def build_neuron(z, w_post, rms_y, v, rho, beta=4.0, gamma=1.0, gate=None, X_on=None):
    """Returns (g_row, u_row, d_col, info). gate=(w, t) for a gated neuron, None for always-on."""
    D = z.shape[0]
    if gate is None:
        g = beta * z
    else:
        w, t = gate
        g = beta * (w - t * z)
    u = gamma * z
    # expected activation on the 'on' class, computed exactly on calibration tokens if given
    if X_on is not None:
        abar = float((gelu_tanh(X_on @ g) * (X_on @ u)).mean())
    else:
        abar = float(gelu_tanh(torch.tensor(beta)) * gamma)
    v_hat = v / v.norm()
    delta = rho * w_post.norm() * v_hat  # same order as the norm of the MLP's normalised write
    # Gemma4RMSNorm multiplies by `weight` (not 1 + weight): dy = rms(y) * delta / w_post, with |w_post| floored.
    w_safe = torch.where(w_post >= 0, 1.0, -1.0) * w_post.abs().clamp_min(1e-3)
    d = rms_y * delta / w_safe / max(abar, 1e-6)
    return g, u, d, dict(abar=abar, delta_norm=float(delta.norm()), d_norm=float(d.norm()))

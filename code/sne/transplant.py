"""Cross-model neuron transplantation (donor: Gemma 4 E4B or 12B; recipient: E2B), closed form, no training.

Both models share the Gemma 4 tokenizer, so they can be run on identical token streams. For a recipient layer l
matched to a donor layer m:
  * input map  M  : x_D ~= M x_R      (ridge, recipient MLP input -> donor MLP input)
  * output map P  : r_R ~= P r_D      (ridge, donor residual after m -> recipient residual after l)
A donor neuron (g_D, u_D, d_D) computes a = gelu(g_D . x_D) (u_D . x_D) and changes the donor residual by
approximately  w_post_D * (a d_D) / rms(y_D). Transplanted into a recycled recipient slot it becomes
  g_R = M^T g_D,  u_R = M^T u_D,  d_R = kappa * rms(y_R) * P (w_post_D * d_D / rms(y_D)) / w_post_R
so that, to first order, it reads the same feature and writes the mapped effect. Fidelity of the read side is
measured directly: correlation between the transplanted activation in the recipient and the donor activation on
held-out tokens.
"""
import torch

from .model import device as _device, layers


@torch.no_grad()
def capture_all(model, tok, pairs, layer_ids, max_tokens_per_seq=192, dev=None, what=("x", "y", "r")):
    """Per token (response tokens only): MLP input x, MLP raw output y, residual after layer r. float16 on CPU."""
    dev = dev or _device()
    L = layers(model)
    bufs = {i: {k: [] for k in what} for i in layer_ids}
    st = {}

    def mk_mlp(i):
        def hook(mod, args, out):
            m = st["mask"]
            if "x" in what:
                bufs[i]["x"].append(args[0][m].half().cpu())
            if "y" in what:
                bufs[i]["y"].append(out[m].half().cpu())
        return hook

    def mk_layer(i):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            bufs[i]["r"].append(h[st["mask"]].half().cpu())
        return hook
    hs = []
    for i in layer_ids:
        if "x" in what or "y" in what:
            hs.append(L[i].mlp.register_forward_hook(mk_mlp(i)))
        if "r" in what:
            hs.append(L[i].register_forward_hook(mk_layer(i)))
    try:
        for p, r in pairs:
            a = tok("<bos>" + p, add_special_tokens=False)["input_ids"]
            b = tok(r + "<turn|>", add_special_tokens=False)["input_ids"][:max_tokens_per_seq]
            ids = torch.tensor([a + b], device=dev)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            mask[0, len(a):] = True
            st["mask"] = mask
            model(input_ids=ids, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return {i: {k: torch.cat(v) for k, v in bufs[i].items()} for i in layer_ids}


def linear_cka(X, Y):
    X = X.float() - X.float().mean(0)
    Y = Y.float() - Y.float().mean(0)
    hsic = (X.T @ Y).pow(2).sum()
    return float(hsic / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))


def ridge(X, Y, lam=1e-2):
    """W minimising ||X W - Y||^2 + lam*tr(X^T X)/D ||W||^2  (X: [N, Dx], Y: [N, Dy]); returns W [Dx, Dy], R^2."""
    X, Y = X.float(), Y.float()
    D = X.shape[1]
    G = X.T @ X
    W = torch.linalg.solve(G + lam * torch.trace(G) / D * torch.eye(D), X.T @ Y)
    res = Y - X @ W
    r2 = 1 - float(res.pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())
    return W, r2


def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def neuron_acts(X, g, u):
    X = X.float()
    return gelu_tanh(X @ g.T) * (X @ u.T)


def map_neurons(gD, uD, dD, Wx, Wr, wpostD, wpostR, rmsD, rmsR, kappa=1.0):
    """gD, uD: [n, D_D]; dD: [D_D, n]. Wx: [D_R, D_D] (x_R @ Wx ~= x_D). Wr: [D_D, D_R] (r_D @ Wr ~= r_R).
    Returns recipient rows/cols gR, uR [n, D_R], dR [D_R, n]."""
    gR = gD.float() @ Wx.T
    uR = uD.float() @ Wx.T
    eff_D = (wpostD.float()[:, None] * dD.float()) / rmsD         # donor residual effect per unit activation
    eff_R = (eff_D.T @ Wr).T                                       # mapped into recipient residual space
    w_safe = torch.where(wpostR >= 0, 1.0, -1.0) * wpostR.abs().clamp_min(1e-3)
    dR = kappa * rmsR * eff_R / w_safe.float()[:, None]
    return gR, uR, dR

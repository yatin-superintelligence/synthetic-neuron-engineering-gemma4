"""Conditional logit edits: a rank-1 change to the last layer's MLP write, built in closed form with no gradient descent.

The last MLP computes y = D a, where a holds the 12288 neuron activations and D is down_proj. The edit is D' = D + u v^T, so
y' = y + u (v . a). Two parts are solved separately:
  * v is a read-out of existing neurons. A weighted ridge regression, solved in closed form, makes v . a about 1 at chosen
    decision positions and about 0 elsewhere. No neuron is replaced, so the rest of the network is untouched.
  * u is the write. The path from y to the logits is RMS norms and scalars, y -> w * y / rms(y) -> s -> w_f * h / rms(h) ->
    logits = E h, so to first order a target logit change along e = E_a - E_b needs
        u = c * rms(y) * rms(h) / s * (e / ||e||^2) / (w * w_f)         (division is elementwise)
    which moves logit_a - logit_b by c at positions where v . a = 1. Saturation is measured, not assumed.
The edit sits after the last attention and MLP, so it can only change the output distribution, and only where v . a is not
small. Both properties are checked on held-out contexts. The delta does not depend on the base weights, so the same LoRA
is exact at every precision.
"""
import numpy as np
import torch

from .model import device as _device, empty_cache as _empty_cache

from .loops import render
from .model import layers, text_model

SQ = '<|"|>'


def edit_decision_points(ids, tok):
    """Indices p where token p+1 is the argument key that follows filepath in an edit_file call."""
    open_id = tok.convert_tokens_to_ids("<|tool_call>")
    head = tok("call:edit_file{filepath:" + SQ, add_special_tokens=False)["input_ids"]
    q = tok.convert_tokens_to_ids(SQ)
    comma = tok(",", add_special_tokens=False)["input_ids"][0]
    out = []
    for i, t in enumerate(ids):
        if t != open_id or ids[i + 1:i + 1 + len(head)] != head:
            continue
        j = i + 1 + len(head)
        while j < len(ids) and ids[j] != q:
            j += 1
        if j + 2 < len(ids) and ids[j + 1] == comma:
            out.append(j + 1)
    return out


def call_text(args):
    """Tool call as the model emits it: file path, then the other arguments in their recorded order."""
    return "<|tool_call>call:edit_file{" + ",".join(f"{k}:{SQ}{v}{SQ}" for k, v in args.items() if isinstance(v, str)) + "}<tool_call|>"


@torch.no_grad()
def gather(model, tok, cases, layer=34, ctx=3000, per_case_neg=300, seed=0, dev=None):
    """Teacher-forced activations at the last MLP. Positives are decision points; negatives are sampled tokens from the same
    sequences and every other token of a tool call, since those are the states most like a decision point."""
    dev = dev or _device()
    Ls = layers(model)
    rng = np.random.default_rng(seed)
    state = {}
    h1 = Ls[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a: state.__setitem__("a", a[0][0].float()))
    h2 = Ls[layer].mlp.register_forward_hook(lambda m, a, o: state.__setitem__("y", o[0].float()))
    h3 = Ls[layer].register_forward_hook(lambda m, a, o: state.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float()))
    open_id = tok.convert_tokens_to_ids("<|tool_call>")
    close_id = tok.convert_tokens_to_ids("<tool_call|>")
    pos, neg, meta, keys, rms_y, rms_h, gaps = [], [], [], [], [], [], []
    old_id = tok("old", add_special_tokens=False)["input_ids"][0]
    new_id = tok("new", add_special_tokens=False)["input_ids"][0]
    try:
        for c in cases:
            text = c["thought"] + "<channel|>" + call_text(c["args"])
            a = tok(render(tok, c), add_special_tokens=False)["input_ids"][-ctx:]
            b = tok(text, add_special_tokens=False)["input_ids"]
            ids = [tok.bos_token_id] + a + b
            dec = [p for p in edit_decision_points(ids, tok)]
            out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=torch.tensor(dec or [0]))
            act, y, h = state["a"], state["y"], state["h"]
            inside, depth = [], False
            for i, t in enumerate(ids):
                depth = depth or t == open_id
                if depth:
                    inside.append(i)
                if t == close_id:
                    depth = False
            pool = np.setdiff1d(np.arange(1, len(ids) - 1), dec)
            take = np.union1d(np.setdiff1d(np.array(inside, dtype=int), dec), rng.choice(pool, min(per_case_neg, len(pool)), replace=False))
            take = take[take < len(ids)]
            neg.append(act[torch.as_tensor(take, device=act.device)].half().cpu())
            meta += [c["instance_id"]] * len(take)
            for p in dec:
                pos.append(act[p].half().cpu())
                keys.append(c["instance_id"])
                rms_y.append(float(y[p].pow(2).mean().sqrt()))
                rms_h.append(float(h[p].pow(2).mean().sqrt()))
                lg = out.logits[0, dec.index(p)]
                gaps.append(float(lg[old_id] - lg[new_id]))
            del out
            _empty_cache()
    finally:
        h1.remove(), h2.remove(), h3.remove()
    return dict(pos=torch.stack(pos), pos_task=keys, neg=torch.cat(neg), neg_task=meta, rms_y=rms_y, rms_h=rms_h, gap=gaps, old_id=old_id, new_id=new_id)


@torch.no_grad()
def gather_loops(model, tok, cases, layer=34, ctx=3000, resp=1500, per_case_neg=300, seed=0, dev=None):
    """Teacher-forced activations for a loop breaker. Positives are sentence ends after loop onset, the points where closing
    the thought is natural. Hard negatives are sentence ends before onset or in normal responses, plus sampled other tokens."""
    dev = dev or _device()
    Ls = layers(model)
    rng = np.random.default_rng(seed)
    state = {}
    h1 = Ls[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a: state.__setitem__("a", a[0][0].float()))
    h2 = Ls[layer].mlp.register_forward_hook(lambda m, a, o: state.__setitem__("y", o[0].float()))
    h3 = Ls[layer].register_forward_hook(lambda m, a, o: state.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float()))
    close = tok.convert_tokens_to_ids("<channel|>")
    pos, neg, hard, pos_task, neg_task, hard_task, rms_y, rms_h, gaps = [], [], [], [], [], [], [], [], []
    top = torch.zeros(tok.vocab_size + 1024)
    try:
        for c in cases:
            a = tok(render(tok, c), add_special_tokens=False)["input_ids"][-ctx:]
            text = c["original_text"]
            b = tok(text, add_special_tokens=False)["input_ids"][:resp]
            if len(b) < 20:
                continue
            onset = len(tok(text[: c["onset_char"]], add_special_tokens=False)["input_ids"]) if c["kind"] == "loop" and c["onset_char"] else None
            ids = [tok.bos_token_id] + a + b
            start = 1 + len(a)
            ends = [i for i in range(start, len(ids) - 1) if tok.decode([ids[i]]).rstrip(" ").endswith((".", "\n", ":"))]
            loop_ends = [i for i in ends if onset is not None and i - start > onset]
            out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=torch.tensor(loop_ends or [0]))
            act, y, h = state["a"], state["y"], state["h"]
            normal_ends = [i for i in ends if i not in set(loop_ends)]
            pool = np.setdiff1d(np.arange(start, len(ids) - 1), ends)
            take = rng.choice(pool, min(per_case_neg, len(pool)), replace=False)
            neg.append(act[torch.as_tensor(take, device=act.device)].half().cpu())
            neg_task += [c["instance_id"]] * len(take)
            if normal_ends:
                hard.append(act[torch.as_tensor(normal_ends, device=act.device)].half().cpu())
                hard_task += [c["instance_id"]] * len(normal_ends)
            for p in loop_ends:
                pos.append(act[p].half().cpu())
                pos_task.append(c["instance_id"])
                rms_y.append(float(y[p].pow(2).mean().sqrt()))
                rms_h.append(float(h[p].pow(2).mean().sqrt()))
                lg = out.logits[0, loop_ends.index(p)].float()
                top[:len(lg)] += torch.softmax(lg, -1).cpu()
                gaps.append(float(lg.max() - lg[close]))
            del out
            _empty_cache()
    finally:
        h1.remove(), h2.remove(), h3.remove()
    return dict(pos=torch.stack(pos), pos_task=pos_task, neg=torch.cat(neg + hard), neg_task=neg_task + hard_task, hard=torch.cat(hard), hard_task=hard_task, rms_y=rms_y, rms_h=rms_h,
                gap=gaps, close_id=close, top_ids=top.topk(50).indices.tolist())


def fit_readout(Xp, Xn, lam=1.0, pos_weight=None):
    """Weighted ridge in the dual: v = X^T S (S K S + lam I)^-1 S t, where lam is relative to the mean diagonal of K."""
    X = torch.cat([Xp, Xn]).double()
    t = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).double()
    w = torch.ones(len(X)).double()
    w[: len(Xp)] = pos_weight or (len(Xn) / max(1, len(Xp)))
    s = w.sqrt()
    Xs = X * s[:, None]
    K = Xs @ Xs.T
    alpha = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype), s * t)
    return (Xs.T @ alpha).float()


def write_vector(model, layer, a_id, b_ids, c, rms_y, rms_h):
    """The u that moves logit_a minus the mean logit of b_ids by c where the read-out equals 1, to first order."""
    tm = text_model(model)
    E = tm.embed_tokens.weight.float().cpu()
    b_ids = [b_ids] if isinstance(b_ids, int) else [b for b in b_ids if b != a_id]
    e = E[a_id] - E[b_ids].mean(0)
    Ls = layers(model)
    w = Ls[layer].post_feedforward_layernorm.weight.float().cpu()
    wf = tm.norm.weight.float().cpu()
    s = float(Ls[layer].layer_scalar.float().cpu())
    return c * rms_y * rms_h / s * (e / e.pow(2).sum()) / (w * wf)


def separation(v, Xp, Xn):
    a, b = (Xp.float() @ v), (Xn.float() @ v)
    if not len(a) or not len(b):  # a tiny preflight slice can leave a split empty
        return dict(pos_mean=0.0, pos_min=0.0, neg_median=0.0, neg_p99=0.0, neg_p999=0.0, neg_max=0.0, n_pos=len(a), n_neg=len(b))
    b = b[torch.randperm(len(b))[:1000000]]  # quantile has a size limit
    q = torch.quantile(b, torch.tensor([0.5, 0.99, 0.999, 1.0]))
    return dict(pos_mean=round(float(a.mean()), 3), pos_min=round(float(a.min()), 3), neg_median=round(float(q[0]), 4), neg_p99=round(float(q[1]), 4),
                neg_p999=round(float(q[2]), 4), neg_max=round(float(q[3]), 4), n_pos=len(a), n_neg=len(b))

"""Human/LLM-readable neuron readouts.

For each selected MLP neuron (layer L, index j):
  * top activating contexts: the k tokens (with 12 tokens of left context) where a_j is largest on a corpus;
  * logit lens of its write: tokens most promoted / suppressed by  W_U . (w_post_L * d_j)  (direction only;
    the post-FFN RMSNorm makes the write's effect proportional to w_post * d_j), excluding special tokens.
"""
import heapq

import torch

from .model import device as _device, empty_cache as _empty_cache

from .model import layers, text_model


@torch.no_grad()
def top_contexts(model, tok, pairs, sel, k=8, ctx=12, dev=None, max_len=768):
    """sel: {L: [j, ...]}. Returns {(L, j): [(act, context_str, token_str), ...]} (top-k by activation)."""
    dev = dev or _device()
    Ls = layers(model)
    heaps = {(L, j): [] for L, js in sel.items() for j in js}
    state = {}

    def mk(L, js):
        idx = torch.tensor(js, device=dev)

        def hook(mod, args):
            a = args[0][0, :, idx].float()  # [T, n]
            m = state["mask"]
            a = a.masked_fill(~m[:, None], float("-inf"))
            vals, pos = a.topk(min(k, a.shape[0]), dim=0)
            for c, j in enumerate(js):
                for v, p in zip(vals[:, c].tolist(), pos[:, c].tolist()):
                    if v == float("-inf"):
                        continue
                    h = heaps[(L, j)]
                    item = (v, state["seq"], p)
                    if len(h) < k:
                        heapq.heappush(h, item)
                    elif v > h[0][0]:
                        heapq.heapreplace(h, item)
        return hook
    hs = [Ls[L].mlp.down_proj.register_forward_pre_hook(mk(L, js)) for L, js in sel.items()]
    seqs = []
    try:
        for s, (p, r) in enumerate(pairs):
            a = tok("<bos>" + p, add_special_tokens=False)["input_ids"]
            b = tok(r + "<turn|>", add_special_tokens=False)["input_ids"]
            ids = (a + b)[:max_len]
            seqs.append(ids)
            t = torch.tensor([ids], device=dev)
            m = torch.zeros(len(ids), dtype=torch.bool, device=dev)
            m[len(a):] = True
            state.update(mask=m, seq=s)
            model(input_ids=t, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    out = {}
    for key, h in heaps.items():
        rows = []
        for v, s, p in sorted(h, reverse=True):
            ids = seqs[s]
            left = tok.decode(ids[max(0, p - ctx):p])
            rows.append((round(v, 3), left.replace("\n", "\\n")[-80:], tok.decode([ids[p]]).replace("\n", "\\n")))
        out[key] = rows
    return out


@torch.no_grad()
def logit_lens(model, tok, sel, k=10):
    tm = text_model(model)
    WU = tm.embed_tokens.weight  # tied embeddings
    Ls = layers(model)
    special = set(tok.all_special_ids)
    out = {}
    for L, js in sel.items():
        wpost = Ls[L].post_feedforward_layernorm.weight.float()
        D = Ls[L].mlp.down_proj.weight[:, js].float()  # [d, n]
        eff = wpost[:, None] * D
        logits = (WU.float() @ eff)  # [V, n]
        for c, j in enumerate(js):
            col = logits[:, c]
            top = [i for i in col.topk(k + 20).indices.tolist() if i not in special][:k]
            bot = [i for i in (-col).topk(k + 20).indices.tolist() if i not in special][:k]
            out[(L, j)] = dict(promotes=[tok.decode([i]) for i in top], suppresses=[tok.decode([i]) for i in bot])
    return out


def atlas_markdown(contexts, lens, scores):
    lines = ["# Neuron atlas", "", "Each entry: layer.index, selectivity scores, top activating tokens (context -> "
             "token, activation) and the vocabulary its write promotes/suppresses (logit lens).", ""]
    for key in contexts:
        L, j = key
        sc = scores.get(key, {})
        lines.append(f"## L{L}.{j}  " + "  ".join(f"{k}={v:.2f}" for k, v in sc.items()))
        ln = lens.get(key, {})
        lines.append(f"- promotes: {ln.get('promotes')}")
        lines.append(f"- suppresses: {ln.get('suppresses')}")
        for v, left, t in contexts[key]:
            lines.append(f"  - `{left}` -> **`{t}`** ({v})")
        lines.append("")
    return "\n".join(lines)

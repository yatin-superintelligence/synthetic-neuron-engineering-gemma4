"""Query-chunked SDPA for attention heads larger than PyTorch's fused kernels support.

Gemma 4's full-attention layers use head_dim 512 (E2B: 7 of 35 layers; 31B likewise), and PyTorch's flash and memory-efficient SDPA
kernels stop at 256, so those layers fall back to the math path, which builds the whole (heads x L x L) score matrix. In training
this ran out of an L4's 22 GB at 13,865 to 19,012 tokens (EXP-110b, 2026-11-10). Splitting the queries into chunks, each recomputed
in the backward pass, keeps the peak near heads x chunk x L. The result is exact: every query gets the same softmax over the same keys.
install() replaces transformers' "sdpa" attention function in place, so masks are built as for SDPA; heads of 256 or less and short
sequences go to the original function unchanged.
"""
import torch
from torch.utils.checkpoint import checkpoint

CHUNK = 1024


def install(chunk=CHUNK, min_head_dim=257, min_len=4096):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(orig, "_sne_chunked", False):
        return orig

    def chunked(module, query, key, value, attention_mask, **kw):
        L, Lk = query.shape[2], key.shape[2]
        if query.shape[-1] < min_head_dim or L <= min_len:
            return orig(module, query, key, value, attention_mask, **kw)
        kw = dict(kw, is_causal=False)
        outs = []
        for s in range(0, L, chunk):
            e = min(L, s + chunk)
            if attention_mask is None:  # causal: query row i sits at absolute position Lk - L + i
                qpos = torch.arange(Lk - L + s, Lk - L + e, device=query.device)[:, None]
                m = (torch.arange(Lk, device=query.device)[None, :] <= qpos)[None, None]
            else:
                m = attention_mask[..., s:e, :]

            def run(q, k, v, m=m):
                return orig(module, q, k, v, m, **kw)[0]

            q = query[:, :, s:e]
            outs.append(checkpoint(run, q, key, value, use_reentrant=False) if torch.is_grad_enabled() else run(q, key, value))
        return torch.cat(outs, dim=1), None

    chunked._sne_chunked = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = chunked
    return chunked

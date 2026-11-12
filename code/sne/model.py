"""Model loading (text-only) on the Intel Arc GPU (torch XPU) and module accessors."""
import os
import time

# Intel Level-Zero caps single allocations at ~4 GB by default; Gemma 4's per-layer-embedding table is 4.7 GB.
os.environ.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import MODELS

DEFAULT = os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized")  # the unused non-QAT checkpoint was removed on 2026-11-10


def device():
    """cuda on Colab, xpu on the laptop's Intel GPU, else cpu. Scripts call this instead of naming a device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


class _CPUEmbedding(torch.nn.Module):
    """Keeps a (huge) embedding table in CPU RAM; looks up on CPU and ships only the result to the device."""

    def __init__(self, inner, dev):
        super().__init__()
        object.__setattr__(self, "inner", inner)  # not registered -> model.to(dev) leaves it on CPU
        self.dev = dev

    def forward(self, ids):
        return self.inner(ids.to("cpu")).to(self.dev)


def load(path=DEFAULT, dtype=torch.bfloat16, dev=None, ple_on_cpu=True, text_only=True, attn_implementation=None):
    dev = dev or device()
    t = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    kw = dict(attn_implementation=attn_implementation) if attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True, **kw)
    if text_only and hasattr(model, "model"):
        # Drop vision/audio towers: they are never used here and cost RAM/GPU memory.
        for name in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
            if hasattr(model.model, name) and getattr(model.model, name) is not None:
                setattr(model.model, name, None)
    ple = None
    tm = text_model(model)
    if ple_on_cpu and getattr(tm, "embed_tokens_per_layer", None) is not None:
        ple = tm.embed_tokens_per_layer
        tm.embed_tokens_per_layer = torch.nn.Identity()
    model.to(dev)  # tensor-by-tensor move avoids one giant warm-up allocation
    if ple is not None:
        tm.embed_tokens_per_layer = _CPUEmbedding(ple, dev)
    model.eval()
    if dev == "xpu":
        import gc
        gc.collect()
    print(f"[load] {type(model).__name__} on {dev} in {time.time()-t:.1f}s; "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.3f}B")
    return model, tok


def text_model(model):
    m = model.model
    return getattr(m, "language_model", m)


def layers(model):
    return text_model(model).layers


def chat_prompts(tok, users, system=None):
    out = []
    for u in users:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out

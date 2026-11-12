"""Compile neuron-level weight edits into exact PEFT LoRA adapters, then into llama.cpp LoRA GGUFs.

An edit is a set of low-rank weight deltas  dW = B @ A  on named linear modules of the text decoder.
Exactness per precision arm: the delta is computed from that arm's own base weights
  * f16 arm  : bf16 safetensors of the QAT full-precision checkpoint,
  * q4_0 arm : the dequantised Q4_0 weights of Google's QAT GGUF,
so that  W_base + dW  equals the intended edited weight for the engine that actually runs it.

Scale convention: PEFT and llama.cpp both use scale = lora_alpha / r. We store a single lora_alpha = 1 and fold
r into B (B_stored = B * r) so that every module, whatever its rank, has effective scale exactly 1.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import MODELS, ROOT

HF_DIR = os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized")
Q4_GGUF = os.path.join(MODELS, "gguf", "gemma-4-E2B_q4_0-it.gguf")
CONVERT_LORA = os.path.join(ROOT, "tools", "llama", "src", "llama.cpp-b12129", "convert_lora_to_gguf.py")

HF_KEY = "model.language_model.layers.{i}.{mod}.weight"
GGUF_KEY = {"mlp.down_proj": "blk.{i}.ffn_down.weight", "mlp.gate_proj": "blk.{i}.ffn_gate.weight",
            "mlp.up_proj": "blk.{i}.ffn_up.weight", "self_attn.o_proj": "blk.{i}.attn_output.weight",
            "self_attn.q_proj": "blk.{i}.attn_q.weight", "self_attn.k_proj": "blk.{i}.attn_k.weight",
            "self_attn.v_proj": "blk.{i}.attn_v.weight", "per_layer_projection": "blk.{i}.proj.weight",
            "per_layer_input_gate": "blk.{i}.inp_gate.weight"}


class BaseWeights:
    """Reads one weight matrix at a time ([out, in], float32) for a precision arm, memory-mapped."""

    def __init__(self, precision, hf_dir=HF_DIR, q4_gguf=Q4_GGUF):
        if precision not in ("f16", "q4_0"):
            raise ValueError(precision)
        self.precision, self._hf_dir, self._q4 = precision, hf_dir, q4_gguf
        self._st = self._rd = self._t = None

    def _open(self):
        """Opened on the first read only. On this Windows laptop an opened 10 GB safetensors file and the 3.2 GB GGUF stayed
        committed as private memory for the whole process (13.2 GB in EXP-089's preflight, 2026-11-08), although edits such as
        rank-1 neurons never read a base weight."""
        if self.precision == "f16" and self._st is None:
            self._st = safe_open(os.path.join(self._hf_dir, "model.safetensors"), "pt")
        elif self.precision == "q4_0" and self._rd is None:
            import gguf
            self._rd = gguf.GGUFReader(self._q4)
            self._t = {t.name: t for t in self._rd.tensors}

    def get(self, i, mod):
        self._open()
        if self.precision == "f16":
            return self._st.get_tensor(HF_KEY.format(i=i, mod=mod)).float()
        import gguf
        t = self._t[GGUF_KEY[mod].format(i=i)]
        w = gguf.quants.dequantize(t.data, t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))  # GGUF stores [out, in] row-major


class Edit:
    """Container of low-rank deltas: {(layer, module): (B [out, r], A [r, in])}."""

    def __init__(self, name, description=""):
        self.name, self.description = name, description
        self.deltas = {}
        self.meta = {}

    def add(self, i, mod, B, A):
        key = (int(i), mod)
        if key in self.deltas:
            B0, A0 = self.deltas[key]
            B, A = torch.cat([B0, B], 1), torch.cat([A0, A], 0)
        self.deltas[key] = (B.float().contiguous(), A.float().contiguous())

    def rank_total(self):
        return sum(A.shape[0] for _, A in self.deltas.values())

    def n_params(self):
        return sum(B.numel() + A.numel() for B, A in self.deltas.values())

    def dense(self, i, mod):
        B, A = self.deltas[(i, mod)]
        return B @ A

    # ---------------------------------------------------------------- export
    def save_peft(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        tensors, rank_pattern, targets = {}, {}, set()
        for (i, mod), (B, A) in sorted(self.deltas.items()):
            r = A.shape[0]
            base = f"base_model.model.model.language_model.layers.{i}.{mod}"
            tensors[f"{base}.lora_A.weight"] = A.to(torch.float32).contiguous()
            tensors[f"{base}.lora_B.weight"] = (B * r).to(torch.float32).contiguous()  # fold r: alpha=1, scale=1/r
            rank_pattern[f"model.language_model.layers.{i}.{mod}"] = r
            targets.add(mod.split(".")[-1])
        rmax = max(rank_pattern.values())
        cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=HF_DIR, r=rmax, lora_alpha=1,
                   rank_pattern=rank_pattern, alpha_pattern={k: 1 for k in rank_pattern},
                   target_modules=sorted(targets), lora_dropout=0.0, bias="none", fan_in_fan_out=False,
                   init_lora_weights=True, inference_mode=True, use_rslora=False)
        save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
        with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(out_dir, "edit_meta.json"), "w", encoding="utf-8") as f:
            json.dump(dict(name=self.name, description=self.description, n_modules=len(self.deltas),
                           rank_total=self.rank_total(), n_params=self.n_params(), meta=self.meta), f, indent=2,
                      default=str)
        return out_dir

    def to_gguf(self, peft_dir, out_path):
        cmd = [sys.executable, CONVERT_LORA, peft_dir, "--base", HF_DIR, "--outtype", "f32", "--outfile", out_path]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError("convert_lora_to_gguf failed:\n" + p.stdout[-2000:] + p.stderr[-3000:])
        return out_path


# ---------------------------------------------------------------- edit primitives
def neuron_scale(edit, W, i, idx, alpha):
    """Scale MLP neurons idx of layer i by alpha (alpha=0 ablates). W = down_proj [d, I] of this arm.
    dW = W[:, idx] (alpha-1) E_idx^T  -> rank |idx|."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    if len(idx) == 0:
        return
    B = W[:, idx] * (alpha - 1.0)
    A = torch.zeros(len(idx), W.shape[1])
    A[torch.arange(len(idx)), idx] = 1.0
    edit.add(i, "mlp.down_proj", B, A)


def neuron_replace(edit, Wg, Wu, Wd, i, slot, g_new, u_new, d_new):
    """Overwrite MLP neuron `slot` of layer i with a new neuron (gate row g_new, up row u_new, down column d_new).
    Each of the three matrices receives a rank-1 delta."""
    e = torch.zeros(1, Wg.shape[0])
    e[0, slot] = 1.0
    edit.add(i, "mlp.gate_proj", e.T.clone(), (g_new - Wg[slot]).view(1, -1))
    edit.add(i, "mlp.up_proj", e.T.clone(), (u_new - Wu[slot]).view(1, -1))
    ed = torch.zeros(1, Wd.shape[1])
    ed[0, slot] = 1.0
    edit.add(i, "mlp.down_proj", (d_new - Wd[:, slot]).view(-1, 1), ed)


def direction_project(edit, W, i, mod, v, lam):
    """Output-space directional edit on a write matrix: W' = (I + lam v v^T) W with unit v (lam=-1 removes v)."""
    v = v / v.norm()
    B = (lam * v).view(-1, 1)
    A = (v.view(1, -1) @ W)
    edit.add(i, mod, B, A)


@torch.no_grad()
def apply_torch(model, edit, sign=1.0):
    """Add (sign=+1) or remove (sign=-1) an Edit's deltas in place on a loaded HF model (for verification)."""
    from .model import layers
    L = layers(model)
    for (i, mod), (B, A) in edit.deltas.items():
        lin = L[i]
        for part in mod.split("."):
            lin = getattr(lin, part)
        lin.weight += sign * (B @ A).to(lin.weight.dtype).to(lin.weight.device)

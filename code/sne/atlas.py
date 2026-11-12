"""Neuron atlas: which MLP neurons respond to which part of real coding work.

Every token of a real agent conversation gets two labels. The segment says what the model is doing: reading the issue, thinking,
writing tool syntax, writing old_string, writing new_string, reading a tool response, and so on. The lexical class says what kind
of code token it is. Activations of all neurons in all layers are summed per label, per half of the tasks, so selectivity is
measured from all tokens with no stored activations, and each result can be checked on tasks it was not measured on.
"""
import keyword
import re

import numpy as np
import torch

from .model import device as _device, empty_cache as _empty_cache

from .model import layers, text_model

SEGMENTS = ["system", "issue", "thought", "call_syntax", "arg_old", "arg_new", "arg_command", "arg_path", "tool_response", "model_text"]
LEXICAL = ["keyword", "identifier", "punctuation", "whitespace", "number", "other"]
CLASSES = SEGMENTS + LEXICAL
CODE_SEGMENTS = ("arg_old", "arg_new", "tool_response")
MIN_TOKENS = 200
LENS_MIN_LAYER = 33
ARG_SEGMENT = {"old_string": "arg_old", "new_string": "arg_new", "command": "arg_command"}


def lexical_class(piece):
    s = piece.strip()
    if not s:
        return "whitespace"
    if keyword.iskeyword(s):
        return "keyword"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        return "identifier"
    if re.fullmatch(r"[0-9.]+", s):
        return "number"
    if re.fullmatch(r"[^\w\s]+", s):
        return "punctuation"
    return "other"


def label_tokens(pieces):
    """Segment and lexical label for each decoded token. Tokens before the first turn count as system."""
    seg, lex = [], []
    region, role, in_string, key, expect_role = "system", "system", False, "", False
    for p in pieces:
        s = region
        if expect_role:
            role = p.strip() or role
            region = {"system": "system", "user": "issue", "model": "model_text"}.get(role, "system")
            s, expect_role = region, False
        elif p == "<|turn>":
            expect_role = True
        elif p == "<turn|>":
            pass
        elif p == "<|channel>":
            region = s = "thought"
        elif p == "<channel|>":
            region = "model_text"
        elif p == "<|tool_call>":
            region, in_string, key = "call", False, ""
            s = "call_syntax"
        elif p == "<tool_call|>":
            region, s = "model_text", "call_syntax"
        elif p == "<|tool_response>":
            region = s = "tool_response"
        elif p == "<tool_response|>":
            region = "model_text"
        elif region == "call":
            if p == '<|"|>':
                in_string = not in_string
                s = "call_syntax"
            elif in_string:
                s = ARG_SEGMENT.get(key.strip(" ,{:"), "arg_path")
            else:
                s = "call_syntax"
                key = "" if p in (",", "{") else key + p
        seg.append(s)
        lex.append(lexical_class(p) if s in CODE_SEGMENTS else None)
    return seg, lex


class Accumulator:
    """Per-class sums of a, a^2 and counts for every neuron of every layer, on the CPU in double precision."""

    def __init__(self, widths):
        c = len(CLASSES)
        self.sum = [np.zeros((c, w)) for w in widths]
        self.sq = [np.zeros((c, w)) for w in widths]
        self.n = np.zeros(c)

    def merged(self, other):
        out = Accumulator([s.shape[1] for s in self.sum])
        for i in range(len(self.sum)):
            out.sum[i] = self.sum[i] + other.sum[i]
            out.sq[i] = self.sq[i] + other.sq[i]
        out.n = self.n + other.n
        return out

    def cohen_d(self, c):
        """One class against all other tokens, per layer."""
        ds = []
        nc, nall = self.n[c], self.n[: len(SEGMENTS)].sum()
        if nc < MIN_TOKENS:  # a class with too few tokens has no meaningful mean, so it reports no selectivity
            return [np.zeros(s.shape[1]) for s in self.sum]
        for s, q in zip(self.sum, self.sq):
            total, total2 = s[: len(SEGMENTS)].sum(0), q[: len(SEGMENTS)].sum(0)
            mc, vc = s[c] / max(1, nc), q[c] / max(1, nc) - (s[c] / max(1, nc)) ** 2
            mr = (total - s[c]) / max(1, nall - nc)
            vr = (total2 - q[c]) / max(1, nall - nc) - mr ** 2
            ds.append((mc - mr) / np.sqrt(np.maximum(0.5 * (vc + vr), 1e-12)))
        return ds


@torch.no_grad()
def collect(model, ids, label_matrix, acc, skip=1, dev=None):
    """One forward pass. label_matrix is [T, classes] of 0 or 1; tokens are read only where the matrix is set."""
    dev = dev or _device()
    Ls = layers(model)
    M = torch.as_tensor(label_matrix, dtype=torch.float32, device=dev)
    hs = []

    def mk(i):
        def hook(mod, args):
            a = args[0][0].float()
            acc.sum[i] += (M.T @ a).double().cpu().numpy()
            acc.sq[i] += (M.T @ (a * a)).double().cpu().numpy()
        return hook
    for i, L in enumerate(Ls):
        hs.append(L.mlp.down_proj.register_forward_pre_hook(mk(i)))
    try:
        model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    acc.n += np.asarray(label_matrix).sum(0)
    _empty_cache()


@torch.no_grad()
def logit_lens(model, tok, neurons, k=8):
    """Tokens most promoted by the combined write of a group's neurons in the last layers. Weights only, so no forward pass.
    A write in an earlier layer passes through later layers before it reaches the logits, so reading it against the embeddings
    gives noise; those neurons are left out, and a group with none reports no lens."""
    neurons = [(i, j) for i, j in neurons if i >= LENS_MIN_LAYER]
    if not neurons:
        return dict(promotes=[], suppresses=[])
    tm = text_model(model)
    E = tm.embed_tokens.weight.float().cpu()
    Ls = layers(model)
    write = torch.zeros(E.shape[1])
    for i, j in neurons:
        write += (Ls[i].post_feedforward_layernorm.weight.float().cpu() * Ls[i].mlp.down_proj.weight[:, j].float().cpu())
    scores = E @ write
    special = set(tok.all_special_ids)
    top = [t for t in scores.topk(k + 30).indices.tolist() if t not in special][:k]
    bottom = [t for t in (-scores).topk(k + 30).indices.tolist() if t not in special][:k]
    return dict(promotes=[tok.decode([t]) for t in top], suppresses=[tok.decode([t]) for t in bottom])

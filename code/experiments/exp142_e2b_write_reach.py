"""EXP-142 (D-181): how far a rank-1 write at E2B's last layer can move the first word of the self-instruction sentence, with the
old write direction and with the minimum-norm one.

Section 8 of the paper says a layer-34 neuron moved "Before" from 14.6 to 5.1 logits behind the most probable token at 32 times its
first-order strength (EXP-118), and D-170 found that the old write direction, e / (w * w_f), is about ten times longer than the
minimum-norm one, e * w * w_f, for the same first-order effect on this token. This measures the write itself, at full read-out
(r = 1), on the states where the sentence would be written: thought starts after a read_file response, development half, the first
24 by order. At each state the write alpha * u_hat is added to layer 34's MLP output at the last position only (a layer-34 write
changes only that position's logits), for both directions and 10 strengths from 0.25 to 1,000 times |y| (the MLP output's norm
there), with e = E["Before"] - mean of E over the five commonest top tokens of the unedited model at these states. Per direction:
the largest margin of "Before" over the best other token at each state, and the number of states where "Before" becomes the most
probable at some strength. One forward pass per state and strength, exact. Runs on the laptop (E2B, bf16 on the Intel GPU).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.model import layers, load, text_model  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.chunked_attention import install  # noqa: E402

DATA = os.path.join(ROOT, "artifacts_local", "EXP-117", "thought_starts_dev.jsonl")
N = 2 if SMOKE else 24
SCALES = [0.25, 1.0] if SMOKE else [0.25, 1, 2, 4, 8, 16, 32, 64, 256, 1000]
MAXTOK = 3000

with Experiment("EXP-142", "e2b-write-reach-old-vs-minnorm", category="diagnosis", depends_on=["EXP-117", "EXP-118", "EXP-140b"],
                motivation="The paper's claim that a last-layer neuron cannot write the sentence's first word was measured with the old "
                           "write direction, which D-170 found about ten times longer than needed for this token.",
                hypothesis="With the minimum-norm direction the first word becomes the most probable at most of the states at some "
                           "strength; with the old direction it does not (EXP-118: 5.1 logits behind at 32 times).",
                config=dict(data="artifacts_local/EXP-117/thought_starts_dev.jsonl (after read_file)", states=N, scales_of_y_norm=SCALES,
                            layer=34, max_tokens=MAXTOK, note="the last 3,000 prompt tokens")) as ex:
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    rows = [r for r in rows if r.get("prev") == "read_file"][:N]
    install()
    model, tok = load()
    dev = next(model.parameters()).device
    L = layers(model)[-1]
    tm = text_model(model)
    before = tok.encode("Before", add_special_tokens=False)
    assert len(before) == 1, before
    B = before[0]
    st = {}
    h_y = L.mlp.register_forward_hook(lambda m, a, o: st.__setitem__("y", o[0, -1].float()))
    add = {"v": None}

    def edit(m, a, o):
        if add["v"] is None:
            return None
        o = o.clone()
        o[0, -1] = (o[0, -1].float() + add["v"]).to(o.dtype)
        return o
    h_e = L.mlp.register_forward_hook(edit)
    ids_all = [tok(r["prompt"], add_special_tokens=False)["input_ids"][-MAXTOK:] for r in rows]

    def run(ids):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[tok.bos_token_id] + ids], device=dev), use_cache=False, logits_to_keep=1)
        return out.logits[0, -1].float()

    base, ynorm, tops = [], [], []
    t0 = time.time()
    for ids in ids_all:
        lg = run(ids)
        base.append(lg.cpu())
        ynorm.append(float(st["y"].norm()))
        tops.append(int(lg.argmax()))
    from collections import Counter
    comp = [t for t, _ in Counter(tops).most_common(6) if t != B][:5]
    ex.log("unedited: top tokens", [(t, tok.decode([t])) for t in comp], "| Before's gap to the top: median",
           round(float(np.median([float(l.max() - l[B]) for l in base])), 2), "| states", len(rows), f"| {time.time() - t0:.0f} s")
    E = tm.embed_tokens.weight
    e = (E[B].float() - E.index_select(0, torch.tensor(comp, device=E.device)).float().mean(0))  # list indexing crashes the laptop GPU driver
    w = L.post_feedforward_layernorm.weight.float()
    wf = tm.norm.weight.float()
    dirs = {"old": e / (w * wf), "minimum_norm": e * w * wf}
    dirs = {k: (v / v.norm()).to(dev) for k, v in dirs.items()}
    out = dict(states=len(rows), competitors=[(t, tok.decode([t])) for t in comp], scales=SCALES,
               unedited_gap_median=round(float(np.median([float(l.max() - l[B]) for l in base])), 3))
    for name, u in dirs.items():
        best_margin, became_top, per_scale = [], 0, {s: [] for s in SCALES}
        for i, ids in enumerate(ids_all):
            mbest = -1e9
            for s in SCALES:
                add["v"] = u * (s * ynorm[i])
                lg = run(ids)
                add["v"] = None
                other = lg.clone()
                other[B] = -1e9
                m = float(lg[B] - other.max())
                per_scale[s].append(m)
                mbest = max(mbest, m)
            best_margin.append(mbest)
            became_top += mbest > 0
        out[name] = dict(states_where_Before_becomes_top=int(became_top), best_margin_median=round(float(np.median(best_margin)), 2),
                         best_margin_min=round(float(min(best_margin)), 2),
                         margin_median_by_scale={str(s): round(float(np.median(v)), 2) for s, v in per_scale.items()})
        ex.log(name, out[name], f"| {time.time() - t0:.0f} s")
        ex.save_json("reach.json", out)
    h_y.remove(), h_e.remove()
    ex.result(reach=out, headline=(f"E2B layer-34 write, 'Before' at {len(rows)} thought starts after a read: old direction top at "
                                   f"{out['old']['states_where_Before_becomes_top']} (best margin median {out['old']['best_margin_median']}), "
                                   f"minimum-norm at {out['minimum_norm']['states_where_Before_becomes_top']} "
                                   f"(best margin median {out['minimum_norm']['best_margin_median']})"))

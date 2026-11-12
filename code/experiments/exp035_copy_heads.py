"""EXP-035: locate the copy circuit of E2B.

Teacher-forced on CopyBench dev verbatim tasks (correct answers) and edit tasks (correct old_string text). For a
response position t, the copy-source set P(t) = {p in prompt : ids[p] == ids[t] and ids[p+1] == ids[t+1]}.
For each layer L and query head h, the copy score is the mean attention weight placed on P(t), and the lift is that
score divided by the score on a control set C(t) = {p in prompt : ids[p] == ids[t], ids[p+1] != ids[t+1]} (same token,
wrong continuation) plus the uniform baseline. High lift means the head selects the right occurrence, the
computational signature of induction/copying (Olsson et al. 2022 for induction heads).
Eager attention on the bf16 reference model. Output: per layer/head table used by later copy-circuit edits.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, copybench  # noqa: E402
from sne.registry import Experiment  # noqa: E402

N_TASKS = 80
with Experiment("EXP-035", "copy-circuit-attention-heads", category="localization", depends_on=["EXP-021", "EXP-027"],
                motivation="H-1 proposes copying as a removable obstacle. Attention heads that select the correct source "
                           "occurrence are the natural substrate for surgical edits (head gain, attention temperature).",
                hypothesis="A small number of heads concentrate the copy signal, mostly in a few layers; many heads have "
                           "lift near 1.",
                config=dict(n_tasks=N_TASKS, tasks="copybench dev verbatim + edit, teacher forced", attn="eager")) as ex:
    from sne.model import load
    model, tok = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"), attn_implementation="eager")
    Ls = model.model.language_model.layers
    nL = len(Ls)
    tasks = [t for t in copybench.load("dev") if t["kind"] in ("verbatim", "edit")][:N_TASKS]
    src = np.zeros((nL, 8))
    ctl = np.zeros((nL, 8))
    uni = np.zeros((nL, 8))
    cnt = 0
    caps = {}

    def mk(i):
        def hook(mod, args, out):
            caps[i] = out[1].detach().float().cpu()
        return hook
    hs = [Ls[i].self_attn.register_forward_hook(mk(i)) for i in range(nL)]
    with torch.no_grad():
        for k, t in enumerate(tasks):
            prompt = "<bos>" + copybench.render(tok, t)
            answer = t["expected"] if t["kind"] == "verbatim" else t["line"]
            a = tok(prompt, add_special_tokens=False)["input_ids"]
            b = tok("```python\n" + answer + "\n```<turn|>", add_special_tokens=False)["input_ids"]
            ids = a + b
            model(input_ids=torch.tensor([ids], device="xpu"), use_cache=False, output_attentions=True)
            T = len(ids)
            pos = [t2 for t2 in range(len(a), T - 1)]
            ready = {}
            for tp in pos:
                P = [p for p in range(1, len(a) - 1) if ids[p] == ids[tp] and ids[p + 1] == ids[tp + 1]]
                C = [p for p in range(1, len(a) - 1) if ids[p] == ids[tp] and ids[p + 1] != ids[tp + 1]]
                if P and C:
                    ready[tp] = (P, C)
            for i in range(nL):
                A = caps[i][0]  # [heads, T, T]
                for tp, (P, C) in ready.items():
                    src[i] += A[:, tp, P].sum(1).numpy()
                    ctl[i] += A[:, tp, C].sum(1).numpy() * (len(P) / len(C))
                    uni[i] += (len(P) / max(1, tp)) * np.ones(A.shape[0])
            cnt += len(ready)
            if k % 20 == 0:
                ex.log(k, "positions", cnt)
    for h in hs:
        h.remove()
    src, ctl, uni = src / max(1, cnt), ctl / max(1, cnt), uni / max(1, cnt)
    lift = src / (ctl + uni + 1e-9)
    np.savez_compressed(os.path.join(ex.dir, "copy_scores.npz"), src=src, ctl=ctl, uni=uni, lift=lift)
    order = np.dstack(np.unravel_index(np.argsort(-(src - ctl).ravel()), src.shape))[0]
    top = [dict(layer=int(L), head=int(h), src=float(src[L, h]), ctl=float(ctl[L, h]), lift=float(lift[L, h]))
           for L, h in order[:24]]
    ex.save_json("top_heads.json", top)
    ex.save_json("layer_summary.json", [dict(layer=i, src_mean=float(src[i].mean()), src_max=float(src[i].max()),
                                             lift_max=float(lift[i].max()), type=model.config.text_config.layer_types[i])
                                        for i in range(nL)])
    ex.result(positions=cnt, top_heads=top[:10], headline=f"top head L{top[0]['layer']}.h{top[0]['head']} src "
              f"{top[0]['src']:.3f} vs ctl {top[0]['ctl']:.3f}; {cnt} copy positions")

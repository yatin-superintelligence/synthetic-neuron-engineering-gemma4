"""EXP-040: is the thinking loop carried by identifiable directions in MLP writes? (dev split of LoopBench only)

Teacher-forced on real traces, PyTorch bf16 reference model, batch size 1. For each LoopBench dev case, the last 3000
prompt tokens plus up to 1500 response tokens. Token classes: LOOP (loop contexts, after onset), PRE (the same responses,
before onset), NORMAL (control contexts). For each layer L, the raw MLP output y (before post_feedforward_layernorm):
  v_L = normalise(mean y | LOOP - mean y | PRE+NORMAL), fitted on half of the dev tasks.
Separation d' of the projection y.v_L is measured on the other half. Layers are ranked by held-out d'.
A real circuit gives high held-out d' in a few layers; a mere token-type artefact gives low or diffuse separation.
Random unit directions give the chance level of d'. Tokens in a loop repeat, so they are not independent samples, and d' is
read against that random level, not against 1. At most 200 tokens per class per case are stored, seeded, and logits are
kept for one position only: a first attempt filled 19 GB of RAM, because a logit table for 4500 tokens is about 2.4 GB
at bf16, and it was stopped before the machine ran out of commit space.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, ROOT, loops as L  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import tokenizer  # noqa: E402

CTX, RESP = 3000, 1500
PER_CLASS = 200
OUT = os.path.join(ROOT, "artifacts_local", "EXP-040")
DIRS = {"exp036": os.path.join(ROOT, "artifacts_local", "kaggle", "exp029", "results", "traces"),
        "exp037": os.path.join(ROOT, "artifacts_local", "kaggle", "exp037", "results", "traces")}

with Experiment("EXP-040", "loop-direction-discovery", category="localization", depends_on=["EXP-038"],
                motivation="If looping is carried by a specific direction written by a few MLP layers, as Takehi et al. "
                           "found for Qwen, it can be removed by a rank-1 edit that compiles exactly to a LoRA.",
                hypothesis="A few layers show held-out d' above 1 and well above random directions; layer ranking is "
                           "stable between task halves.",
                config=dict(ctx_tokens=CTX, resp_tokens=RESP, tokens_per_class=PER_CLASS, split="dev", fit_half_seed=40, classes=["LOOP", "PRE", "NORMAL"])) as ex:
    tok = tokenizer()
    L.build(DIRS)  # both official runs, with the dev and test split shared by every replay benchmark
    cases = L.load("dev")
    tasks = sorted({c["instance_id"] for c in cases})
    random.Random(40).shuffle(tasks)
    fit_tasks = set(tasks[: len(tasks) // 2])
    ex.log("cases", len(cases), "loop", sum(c["kind"] == "loop" for c in cases), "tasks", len(tasks))
    from sne.model import load
    model, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    Ls = model.model.language_model.layers
    nL = len(Ls)
    D = model.config.text_config.hidden_size
    state = {}

    def mk(i):
        def hook(mod, args, out):
            state.setdefault("y", {})[i] = out[0, state["start"]:].half().cpu()
        return hook
    hs = [Ls[i].mlp.register_forward_hook(mk(i)) for i in range(nL)]
    t0 = time.time()
    rng = np.random.default_rng(40)
    store = {"fit": {i: {"LOOP": [], "OTHER": []} for i in range(nL)}, "hold": {i: {"LOOP": [], "OTHER": []} for i in range(nL)}}
    with torch.no_grad():
        for k, c in enumerate(cases):
            a = tok(L.render(tok, c), add_special_tokens=False)["input_ids"][-CTX:]
            text = c["original_text"]
            b = tok(text, add_special_tokens=False)["input_ids"][:RESP]
            if len(b) < 20:
                continue
            onset = len(tok(text[: c["onset_char"]], add_special_tokens=False)["input_ids"]) if c["kind"] == "loop" and c["onset_char"] else None
            state.clear()
            state["start"] = 1 + len(a)
            model(input_ids=torch.tensor([[tok.bos_token_id] + a + b], device="xpu"), use_cache=False, logits_to_keep=1)
            part = "fit" if c["instance_id"] in fit_tasks else "hold"
            pos = np.arange(len(b))
            if onset is not None:
                loop_pos, other_pos = pos[onset:], pos[:onset]
            else:
                loop_pos, other_pos = pos[:0], pos
            loop_pos = rng.permutation(loop_pos)[:PER_CLASS]
            other_pos = rng.permutation(other_pos)[:PER_CLASS]
            for i in range(nL):
                y = state["y"][i]
                if len(loop_pos):
                    store[part][i]["LOOP"].append(y[loop_pos].clone())
                if len(other_pos):
                    store[part][i]["OTHER"].append(y[other_pos].clone())
            torch.xpu.empty_cache()
            if k % 10 == 0:
                ex.log(k, f"{time.time() - t0:.0f}s")
    for h in hs:
        h.remove()
    del model
    rows = []
    vs = {}
    torch.manual_seed(40)
    for i in range(nL):
        fl, fo = torch.cat(store["fit"][i]["LOOP"]).float(), torch.cat(store["fit"][i]["OTHER"]).float()
        hl, ho = torch.cat(store["hold"][i]["LOOP"]).float(), torch.cat(store["hold"][i]["OTHER"]).float()
        v = fl.mean(0) - fo.mean(0)
        gap_fit = float(v.norm())
        v = v / v.norm()

        def dp(u):
            a, b = hl @ u, ho @ u
            return float((a.mean() - b.mean()) / (0.5 * (a.var() + b.var())).sqrt().clamp_min(1e-8))
        r = torch.randn(32, D)
        rnd = [abs(dp(x / x.norm())) for x in r]
        # reverse half gives the reliability of the direction itself
        v2 = hl.mean(0) - ho.mean(0)
        cos = float((v @ v2) / v2.norm())
        rows.append(dict(layer=i, d_prime_heldout=round(dp(v), 3), d_prime_fit_gap=round(gap_fit, 4), random_abs_d_max=round(max(rnd), 3),
                         random_abs_d_mean=round(float(np.mean(rnd)), 3), cosine_between_halves=round(cos, 3),
                         n_loop_tokens=int(fl.shape[0] + hl.shape[0]), n_other_tokens=int(fo.shape[0] + ho.shape[0])))
        vs[i] = v
        ex.log(rows[-1])
    os.makedirs(OUT, exist_ok=True)
    torch.save(vs, os.path.join(OUT, "directions.pt"))
    ex.save_json("layers.json", rows)
    top = sorted(rows, key=lambda r: -abs(r["d_prime_heldout"]))[:6]
    ex.result(top_layers=top, headline="top held-out d': " + ", ".join(f"L{r['layer']} {r['d_prime_heldout']} (cos {r['cosine_between_halves']})" for r in top[:4]) +
                                       f"; random max {max(r['random_abs_d_max'] for r in rows):.2f}")

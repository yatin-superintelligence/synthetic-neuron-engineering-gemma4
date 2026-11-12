"""EXP-068: what does E2B's state look like when it continues the way E4B does, instead of looping? (dev split only)

EXP-071 put E4B into E2B's own loop states, and E4B left every one of them (0 of 78 replays looped). That gives, for the first time,
a non-looping continuation of the exact same state. EXP-040's direction contrasted loop tokens with tokens from other moments, so it
mixes looping with lateness, repetition and context. Here the contrast is tighter: E2B is teacher-forced on the same prompt twice,
once with its own looping continuation and once with E4B's continuation, and the MLP outputs are compared position by position class.
Only thought text is used on both sides, so the contrast is not "about to call a tool" against "thinking".

Classes, per dev loop state: LOOP (E2B's own continuation after loop onset), PRE (before onset), TEACHER (E4B's thought text, both
seeds and both precisions from EXP-071, up to 4 continuations). Directions v_T = normalise(mean LOOP - mean TEACHER) per layer, fitted on
the same half of dev tasks as EXP-040 (seed 40), measured on the other half against random directions.

Decision rule, written before the run. If v_T has cosine of at least 0.9 with EXP-040's direction at layers 14, 15, 19 and 23, both
methods found the same axis: that is convergent evidence for the paper and needs no new edit. If cosine is below 0.9 at two or more of
those layers, and v_T separates held-out LOOP from TEACHER better than EXP-040's direction does, v_T becomes a new edit, tested on the
test split at both precisions in a separate experiment. No test-split text is read here.
Precision: this measures directions on the bf16 reference model, as EXP-040 did. The edit it may lead to is tested at Q4_0 and F16.
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
EDIT_LAYERS = (14, 15, 19, 23)
OUT = os.path.join(ROOT, "artifacts_local", "EXP-068")
TEACHER = [os.path.join(ROOT, "artifacts_local", "EXP-071", f"texts_{p}.json") for p in ("q4_0", "f16")]
OLD = os.path.join(ROOT, "artifacts_local", "EXP-040", "directions.pt")

with Experiment("EXP-068", "e4b-contrast-direction", category="localization", depends_on=["EXP-040", "EXP-071"],
                motivation="E4B leaves the states that trap E2B. Contrasting E2B's state on its own loop with its state on E4B's continuation of the same prompt isolates looping from lateness and context.",
                hypothesis="v_T agrees with EXP-040's direction (cosine at least 0.9 at the edit layers). If not, it separates held-out loop from teacher tokens better and becomes a new edit.",
                config=dict(ctx_tokens=CTX, resp_tokens=RESP, tokens_per_class=PER_CLASS, split="dev", fit_half_seed=40,
                            classes=["LOOP", "PRE", "TEACHER"], edit_layers=EDIT_LAYERS, cosine_rule=0.9)) as ex:
    tok = tokenizer()
    teacher = {}
    for p in TEACHER:
        if os.path.exists(p):
            for k, v in json.load(open(p, encoding="utf-8")).items():
                teacher.setdefault(k.split("#")[0], []).append(v.split("<channel|>")[0])
    cases = L.load("dev")
    tasks = sorted({c["instance_id"] for c in cases})
    random.Random(40).shuffle(tasks)
    fit_tasks = set(tasks[: len(tasks) // 2])  # the same halves as EXP-040
    loops_only = [c for c in cases if c["kind"] == "loop" and teacher.get(c["case"])]
    ex.log("dev loop states with E4B continuations", len(loops_only), "teacher texts", sum(len(teacher[c["case"]]) for c in loops_only),
           "fit tasks", len(fit_tasks))
    if not loops_only:
        raise RuntimeError("no states with teacher text")
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
    rng = np.random.default_rng(68)
    store = {part: {i: {"LOOP": [], "PRE": [], "TEACHER": []} for i in range(nL)} for part in ("fit", "hold")}
    t0 = time.time()

    def forward(a, b):
        state.clear()
        state["start"] = 1 + len(a)
        model(input_ids=torch.tensor([[tok.bos_token_id] + a + b], device="xpu"), use_cache=False, logits_to_keep=1)
        return state["y"]

    with torch.no_grad():
        for k, c in enumerate(loops_only):
            part = "fit" if c["instance_id"] in fit_tasks else "hold"
            a = tok(L.render(tok, c), add_special_tokens=False)["input_ids"][-CTX:]
            text = c["original_text"]
            b = tok(text, add_special_tokens=False)["input_ids"][:RESP]
            onset = len(tok(text[: c["onset_char"]], add_special_tokens=False)["input_ids"]) if c.get("onset_char") else 0
            y = forward(a, b)
            pos = np.arange(len(b))
            for name, sel in (("LOOP", pos[onset:]), ("PRE", pos[:onset])):
                sel = rng.permutation(sel)[:PER_CLASS]
                if len(sel):
                    for i in range(nL):
                        store[part][i][name].append(y[i][sel].clone())
            for t in teacher[c["case"]]:
                b = tok(t, add_special_tokens=False)["input_ids"][:RESP]
                if len(b) < 10:
                    continue
                y = forward(a, b)
                sel = rng.permutation(np.arange(len(b)))[:PER_CLASS]
                for i in range(nL):
                    store[part][i]["TEACHER"].append(y[i][sel].clone())
            torch.xpu.empty_cache()
            ex.log(k, c["case"], part, f"{time.time() - t0:.0f}s")
    for h in hs:
        h.remove()
    del model
    old = torch.load(OLD) if os.path.exists(OLD) else {}
    torch.manual_seed(68)
    rows, vs = [], {}

    def cat(part, i, name):
        x = store[part][i][name]
        return torch.cat(x).float() if x else torch.zeros(0, D)

    def dprime(u, x, y):
        if not len(x) or not len(y):
            return None
        a, b = x @ u, y @ u
        return round(float((a.mean() - b.mean()) / (0.5 * (a.var() + b.var())).sqrt().clamp_min(1e-8)), 3)

    for i in range(nL):
        fl, ft = cat("fit", i, "LOOP"), cat("fit", i, "TEACHER")
        hl, ht, hp = cat("hold", i, "LOOP"), cat("hold", i, "TEACHER"), cat("hold", i, "PRE")
        if not len(fl) or not len(ft):
            continue
        v = fl.mean(0) - ft.mean(0)
        v = v / v.norm()
        vs[i] = v
        r = torch.randn(32, D)
        rnd = [abs(dprime(x / x.norm(), hl, ht) or 0) for x in r]
        u = old.get(i)
        rows.append(dict(layer=i, d_loop_vs_teacher_heldout=dprime(v, hl, ht), random_abs_d_max=round(max(rnd), 3),
                         old_direction_d_loop_vs_teacher=dprime(u.float(), hl, ht) if u is not None else None,
                         new_direction_d_loop_vs_pre=dprime(v, hl, hp), cosine_with_exp040=round(float(v @ u.float()), 3) if u is not None else None,
                         n_tokens=dict(fit_loop=len(fl), fit_teacher=len(ft), hold_loop=len(hl), hold_teacher=len(ht), hold_pre=len(hp))))
        ex.log(rows[-1])
    os.makedirs(OUT, exist_ok=True)
    torch.save(vs, os.path.join(OUT, "directions.pt"))
    ex.save_json("layers.json", rows)
    at = {r["layer"]: r for r in rows if r["layer"] in EDIT_LAYERS}
    below = [l for l, r in at.items() if (r["cosine_with_exp040"] or 0) < 0.9]
    better = [l for l, r in at.items() if (r["d_loop_vs_teacher_heldout"] or 0) > (r["old_direction_d_loop_vs_teacher"] or 0)]
    decision = "same axis as EXP-040" if len(below) < 2 else ("new edit to test" if len(set(below) & set(better)) >= 2 else "different but not better")
    ex.result(edit_layers=at, decision=decision,
              headline=f"{decision}; cosine with EXP-040 at layers 14/15/19/23: " + ", ".join(str(at[l]["cosine_with_exp040"]) for l in EDIT_LAYERS if l in at)
              + "; held-out d' loop vs E4B: " + ", ".join(f"new {at[l]['d_loop_vs_teacher_heldout']} old {at[l]['old_direction_d_loop_vs_teacher']}" for l in EDIT_LAYERS if l in at))

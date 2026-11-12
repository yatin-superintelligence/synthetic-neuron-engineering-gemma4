"""EXP-082: when does E2B decide which tool to call, before or during its thinking?

EXP-080 found that at the tool-name token E2B's choice is usually settled (31B's alternative trails by 7 to 12 logits at its
positives), so a neuron that writes there turns few decisions. This measures where in the turn the upcoming action becomes readable
from E2B's own activations, which says where a decision neuron should write.

Fixed before the run:
- States: the 1,597 tool-call states of the three unedited base runs (exp029, exp037, exp-044a), full context as in EXP-079/080 (cut
  to 22,000 tokens), followed by E2B's own thinking from the trace and <channel|><|tool_call>call:.
- Positions: the first thinking token (before any thinking), 25%, 50% and 75% of the way through the thinking, and the tool-name
  position (the last token). States with fewer than 8 thinking tokens are kept, with the fractional positions rounded.
- Features: the activation entering layer 34's down_proj (12,288 values) and the residual stream after layer 17 (1,536 values).
- Labels: E2B's own next action, among the six most common tools (others dropped), and 31B's majority action (EXP-079).
- Probe: one-against-rest ridge in closed form (the same fit_readout as the neurons, ridge 1), fitted on dev-split states, scored on
  test-split states: accuracy of the arg-max, against the majority-class rate. Also the logit lens at layer 34 for comparison: the
  share of test states where the actual tool's first token is the top tool token at each position.
Prediction written before the run: E2B's own action is far more readable at the tool-name position than at the first thinking token
(at least 20 accuracy points more), and 31B's majority action is about as readable at the first thinking token as E2B's own.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, ROOT, SMOKE  # noqa: E402
from sne.conditional import fit_readout  # noqa: E402
from sne.model import device, empty_cache, layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import tokenizer  # noqa: E402
import exp079_teacher_decisions as T  # noqa: E402

LAYER_MLP, LAYER_RES, LAM = 34, 17, 1.0
FRACS = ("start", "q25", "q50", "q75", "call")
TOOLS6 = ["read_file", "edit_file", "search_similar_code", "run_command", "submit_patch", "write_file"]
LABELS = os.path.join(ROOT, "artifacts_local", "EXP-079", "teacher_labels.jsonl")


@torch.no_grad()
def gather(model, tok, rows, dev, tool_ids):
    Ls = layers(model)
    st = {}
    h1 = Ls[LAYER_MLP].mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("mlp", a[0][0].float()))
    h2 = Ls[LAYER_RES].register_forward_hook(lambda m, a, o: st.__setitem__("res", (o[0] if isinstance(o, tuple) else o)[0].float()))
    cache = {}
    try:
        for k, r in enumerate(rows):
            t = cache.setdefault(r["trace"], json.load(open(r["trace"], encoding="utf-8")))
            ids, _ = T.prompt(tok, t, r["step"])
            think = tok(t["steps"][r["step"]].get("message") or "", add_special_tokens=False)["input_ids"]
            tail = tok("<channel|><|tool_call>call:", add_special_tokens=False)["input_ids"]
            full = [tok.bos_token_id] + ids + think + tail
            p0 = 1 + len(ids)  # first thinking token (or the closing tokens when there is no thinking)
            n = len(think)
            pos = dict(start=p0, q25=p0 + int(0.25 * n), q50=p0 + int(0.5 * n), q75=p0 + int(0.75 * n), call=len(full) - 1)
            keep = sorted(set(pos.values()))
            out = model(input_ids=torch.tensor([full], device=dev), use_cache=False, logits_to_keep=torch.tensor(keep, device=dev))
            lg = out.logits[0]
            row = {p: i for i, p in enumerate(keep)}
            r["mlp"] = {f: st["mlp"][p].cpu().half() for f, p in pos.items()}
            r["res"] = {f: st["res"][p].cpu().half() for f, p in pos.items()}
            # the model's own output logits at each position: which tool's first token ranks highest among the six
            r["lens"] = {f: max(tool_ids, key=lambda tn: float(lg[row[p], tool_ids[tn]])) for f, p in pos.items()}
            r["think_tokens"] = n
            del out, lg
            if k % 50 == 0:
                empty_cache()
                print(f"[gather] {k}/{len(rows)}", flush=True)
    finally:
        h1.remove(), h2.remove()


def probe_acc(fit, test, key, frac, label):
    classes = sorted({r[label] for r in fit})
    W = []
    for c in classes:
        P = torch.stack([r[key][frac].float() for r in fit if r[label] == c])
        N = torch.stack([r[key][frac].float() for r in fit if r[label] != c])
        W.append(fit_readout(P, N, lam=LAM))
    W = torch.stack(W, 1)
    X = torch.stack([r[key][frac].float() for r in test])
    pred = [classes[i] for i in X.matmul(W).argmax(1).tolist()]
    return round(float(np.mean([p == r[label] for p, r in zip(pred, test)])), 3)


with Experiment("EXP-082", "decision-timing", category="interpretability", depends_on=["EXP-079", "EXP-080"],
                motivation="EXP-080's neurons write at the tool-name token, where E2B's choice is already settled; find where the choice forms.",
                hypothesis="E2B's own action is far more readable at the tool-name position than at the first thinking token (at least 20 points).",
                config=dict(layer_mlp=LAYER_MLP, layer_res=LAYER_RES, positions=FRACS, tools=TOOLS6, ridge=LAM, max_prompt=T.MAX_PROMPT)) as ex:
    labels = {(r["run"], r["iid"], r["step"]): r for r in map(json.loads, open(LABELS, encoding="utf-8"))}
    rows = [r for r in T.states() if r["e2b"] in TOOLS6]
    for r in rows:
        lab = labels.get((r["run"], r["iid"], r["step"]))
        r["t_major"] = collections.Counter(lab["teacher"]).most_common(1)[0][0] if lab else None
    if SMOKE:
        rows = rows[:16]
    ex.log("states", len(rows), collections.Counter(r["e2b"] for r in rows).most_common())
    tok = tokenizer()
    tool_ids = {}
    a = tok("<|tool_call>call:", add_special_tokens=False)["input_ids"]
    for n in TOOLS6:
        tool_ids[n] = tok("<|tool_call>call:" + n, add_special_tokens=False)["input_ids"][len(a)]
    from sne.model import load
    model, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    dev = device()
    gather(model, tok, rows, dev, tool_ids)
    fit = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    out = {}
    for label, sub in (("e2b", lambda r: True), ("t_major", lambda r: r["t_major"] in TOOLS6)):
        f2, t2 = [r for r in fit if sub(r)], [r for r in test if sub(r)]
        base = collections.Counter(r[label] for r in t2).most_common(1)[0][1] / len(t2)
        res = {"majority_class_rate": round(base, 3), "n_test": len(t2)}
        for key in ("mlp", "res"):
            res[key] = {fr: probe_acc(f2, t2, key, fr, label) for fr in FRACS}
        if label == "e2b":
            res["logit_lens_top_tool_matches"] = {fr: round(float(np.mean([r["lens"][fr] == r["e2b"] for r in t2])), 3) for fr in FRACS}
        out[label] = res
        ex.log(label, res)
    ex.result(**out, think_tokens_median=int(np.median([r["think_tokens"] for r in rows])),
              headline=f"E2B's own action from layer-34 MLP activations: " + ", ".join(f"{fr} {out['e2b']['mlp'][fr]}" for fr in FRACS)
                       + f" (majority class {out['e2b']['majority_class_rate']}); 31B's action: " + ", ".join(f"{fr} {out['t_major']['mlp'][fr]}" for fr in FRACS))

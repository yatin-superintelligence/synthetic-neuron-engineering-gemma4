"""EXP-081: does EXP-076's re-read neuron still fire when E2B sees its whole context, as it does on real tasks?

EXP-076 fitted and checked the N_reread read-out on activations computed from only the last 3,000 tokens of each state. On real tasks
the agent sees its whole (compacted) context, about 10,000 tokens. In the first real-task run (EXP-076a, seed 1) the share of failed
edits followed by read_file was 30 of 67 (45%), inside the unedited range (31% to 46% in five runs), so the neuron did not change the
decision as designed. This measures the read-out on the same states at full context (cut to 22,000 tokens as in EXP-079), against the
3,000-token numbers, and the logit gap read minus edit there, so the expected flip rate under the +3 write can be computed.

Fixed before the run: all N_reread positives (both splits) and 300 negatives drawn with seed 81 from the EXP-076 state list; bf16
reference E2B; layer 34. Prediction written before the run: if the positives' mean read-out at full context is below 0.35, the
context mismatch explains the null mechanism result and EXP-076 must be refitted at full context before more real-task runs.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, ROOT, SMOKE  # noqa: E402
from sne.conditional import separation  # noqa: E402
from sne.model import device, empty_cache, layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import tokenizer  # noqa: E402
import exp079_teacher_decisions as T  # noqa: E402

LAYER, CTX76 = 34, 3000
READ, EDIT = 1399, 8982


def failed(obs):
    t = obs.get("content") if isinstance(obs, dict) else json.dumps(obs or "")
    return isinstance(t, str) and '"status": "error"' in t.replace("'", '"')


def reread_rows():
    rows = []
    for r in T.states():
        if r["e2b"] == "STOP":
            continue
        rows.append(r)
    cache = {}
    for r in rows:
        t = cache.setdefault(r["trace"], json.load(open(r["trace"], encoding="utf-8")))
        prev = [s for j, s in T.L.agent_steps(t) if j < r["step"] and s.get("tool_calls")]
        r["reread"] = bool(prev) and prev[-1]["tool_calls"][0].get("function_name") == "edit_file" and failed(prev[-1].get("observation"))
    return rows


@torch.no_grad()
def act(model, tok, t, r, dev, full):
    Ls = layers(model)
    st = {}
    h = Ls[LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0, -1].float().cpu()))
    try:
        msg = (t["steps"][r["step"]].get("message") or "") + "<channel|><|tool_call>call:"
        if full:
            ids, _ = T.prompt(tok, t, r["step"])
            ids = ids + tok(msg, add_special_tokens=False)["input_ids"]
        else:  # exactly as EXP-076 built its rows
            ids = tok(T.L.render(tok, dict(messages=T.L.messages_before(t, r["step"]))) + msg, add_special_tokens=False)["input_ids"][-CTX76:]
        out = model(input_ids=torch.tensor([[tok.bos_token_id] + ids], device=dev), use_cache=False, logits_to_keep=1)
        lg = out.logits[0, -1].float().cpu()
        return st["a"], float(lg[READ] - lg[EDIT]), len(ids) + 1
    finally:
        h.remove()


with Experiment("EXP-081", "reread-full-context", category="diagnostic", depends_on=["EXP-076"],
                motivation="EXP-076a showed no change in the re-read decision on real tasks; the read-out was fitted on 3,000-token contexts.",
                hypothesis="At full context the N_reread read-out on its positives is far lower than at 3,000 tokens (below 0.35).",
                config=dict(layer=LAYER, ctx_exp076=CTX76, max_prompt=T.MAX_PROMPT, negatives=300, seed=81)) as ex:
    rows = reread_rows()
    pos = [r for r in rows if r["reread"]]
    rng = random.Random(81)
    neg = rng.sample([r for r in rows if not r["reread"]], 300)
    if SMOKE:
        pos, neg = pos[:4], neg[:4]
    ex.log("positives", len(pos), "negatives", len(neg))
    v = torch.load(os.path.join(ROOT, "artifacts_local", "EXP-076", "neuron_reread.pt"))["v"].float()
    tok = tokenizer()
    from sne.model import load
    model, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    dev = device()
    cache = {}
    res = {}
    for full in (False, True):
        key = "full" if full else "last3000"
        A = {}
        for k, r in enumerate(pos + neg):
            t = cache.setdefault(r["trace"], json.load(open(r["trace"], encoding="utf-8")))
            a, gap, n = act(model, tok, t, r, dev, full)
            A[id(r)] = (a, gap, n)
            if k % 50 == 0:
                empty_cache()
                print(f"[{key}] {k}/{len(pos) + len(neg)}", flush=True)
        P = torch.stack([A[id(r)][0] for r in pos])
        N = torch.stack([A[id(r)][0] for r in neg])
        ro = P.matmul(v).numpy()
        gaps = np.array([A[id(r)][1] for r in pos])
        res[key] = dict(separation=separation(v, P, N), tokens_median=int(np.median([A[id(r)][2] for r in pos])),
                        gap_read_minus_edit_median=round(float(np.median(gaps)), 3),
                        flip_share_if_shift_is_3_times_readout=round(float(np.mean(gaps + 3.0 * ro > 0)), 3),
                        read_preferred_already=round(float(np.mean(gaps > 0)), 3))
        ex.log(key, res[key])
    ex.result(**res, headline=f"N_reread positives: read-out {res['last3000']['separation']['pos_mean']} at 3,000 tokens, "
                              f"{res['full']['separation']['pos_mean']} at full context; predicted flip share "
                              f"{res['last3000']['flip_share_if_shift_is_3_times_readout']} vs {res['full']['flip_share_if_shift_is_3_times_readout']}")

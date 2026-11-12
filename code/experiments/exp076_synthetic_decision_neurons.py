"""EXP-076: two synthetic decision neurons for the E2B agent, built in closed form from official traces, with no gradient training.

A synthetic neuron here is a new rank-1 unit in the last MLP (layer 34): a read-out v over the 12,288 existing neuron activations that
is about 1 at one kind of agent decision and about 0 elsewhere, and a write u that shifts the next-token logits at that decision only.
D' = D + u v^T, an exact rank-1 LoRA (src/sne/conditional.py has the derivation).

N_reread. Where: the first token of the tool name, right after the agent's edit_file call failed. In the three base runs, 168
edit_file calls failed; 118 were followed later by another edit_file, which resent the same old_string in 69 cases, and 1 of those
69 applied (recounted 2026-11-07 23:40 UTC with tools/decision_metrics.py's definitions; an earlier draft said 141 of 209, a count
that could not be reproduced). Write: raise "read" over "edit".
N_test. Where: the first token of the tool name, when at least one edit has applied since the last run_command. The agent submits
patches that fail tests in 36 of 58 tasks (EXP-044a). Write: raise "run" over "submit". Correction found at the first build
(2026-11-07 23:20 UTC), before any real-task run: the agent already runs pytest after its last applied edit in 32 of 50 such tasks
(EXP-044a); at N_test's positions the base model prefers "run" over "submit" by 8.9 logits, so a shift of 3 cannot change its
choice; and 47 of 58 tasks end on a step with no tool call, a decision this position does not reach. N_test is kept for the record
and is expected to do nothing; the real-task test is built around N_reread. The stopping decision needs its own neuron (EXP-078).
Control: for each neuron, a read-out fitted to the same number of randomly chosen tool-name positions, with the same write.

Fixed before the run: layer 34, logit shift c = 3 at full read-out (about 20 times the odds), ridge strength 1, fitting on the dev
split of tasks only (split seed 20261106), read-out separation checked on the test split, which is not used to fit. Traces come from
the three base E2B official runs (exp029, exp037, exp-044a). The real-task test runs afterwards on Colab (EXP-076 run list).
Precision: built on the bf16 reference model. The delta does not depend on base weights, so the same adapter is exact at every
precision; the real-task test is on the official W4A16 model.
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, ROOT, SMOKE, loops as L  # noqa: E402
from sne.conditional import fit_readout, separation, write_vector  # noqa: E402
from sne.model import device, empty_cache, layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import tokenizer  # noqa: E402

LAYER, C, LAM, CTX = 34, 3.0, 1.0, 3000
RUNS = {r: os.path.join(ROOT, "artifacts_local", "kaggle", r, "results", "traces") for r in ("exp029", "exp037", "exp-044a")}
NAMES = dict(read=1399, edit=8982, run=4503, submit=15037)
SQ = '<|"|>'
OUT = os.path.join(ROOT, "artifacts_local", "EXP-076")


def failed(obs):
    t = obs.get("content") if isinstance(obs, dict) else json.dumps(obs or "")
    return isinstance(t, str) and '"status": "error"' in t.replace("'", '"')


def states():
    """Every agent step that made a tool call, labelled for both neurons."""
    split = L.split_map()
    rows = []
    for run, d in RUNS.items():
        for f in sorted(glob.glob(os.path.join(d, "trace_*.json"))):
            iid = os.path.basename(f)[6:-5]
            t = json.load(open(f, encoding="utf-8"))
            prev_failed_edit, edited_untested = False, False
            for i, s in L.agent_steps(t):
                calls = s.get("tool_calls") or []
                if calls:
                    rows.append(dict(run=run, iid=iid, step=i, split=split.get(iid, "dev"), trace=f, name=calls[0].get("function_name"),
                                     reread=prev_failed_edit, test=edited_untested))
                    name = calls[0].get("function_name")
                    bad = failed(s.get("observation"))
                    prev_failed_edit = name == "edit_file" and bad
                    if name in ("edit_file", "write_file") and not bad:
                        edited_untested = True
                    if name == "run_command":
                        edited_untested = False
    return rows


@torch.no_grad()
def gather(model, tok, rows, dev):
    """Activation entering down_proj at layer 34, and the RMS terms the write needs, at the token before the tool name."""
    Ls = layers(model)
    st = {}
    h1 = Ls[LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0, -1].float().cpu()))
    h2 = Ls[LAYER].mlp.register_forward_hook(lambda m, a, o: st.__setitem__("y", o[0, -1].float()))
    h3 = Ls[LAYER].register_forward_hook(lambda m, a, o: st.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
    cache = {}
    try:
        for k, r in enumerate(rows):
            t = cache.setdefault(r["trace"], json.load(open(r["trace"], encoding="utf-8")))
            s = t["steps"][r["step"]]
            case = dict(messages=L.messages_before(t, r["step"]))
            prefix = L.render(tok, case) + (s.get("message") or "") + "<channel|><|tool_call>call:"
            ids = tok(prefix, add_special_tokens=False)["input_ids"][-CTX:]
            out = model(input_ids=torch.tensor([[tok.bos_token_id] + ids], device=dev), use_cache=False, logits_to_keep=1)
            lg = out.logits[0, -1].float().cpu()
            r.update(a=st["a"].half(), rms_y=float(st["y"].pow(2).mean().sqrt()), rms_h=float(st["h"].pow(2).mean().sqrt()),
                     gap_read_edit=float(lg[NAMES["read"]] - lg[NAMES["edit"]]), gap_run_submit=float(lg[NAMES["run"]] - lg[NAMES["submit"]]))
            del out
            if k % 25 == 0:
                empty_cache()
                print(f"[gather] {k}/{len(rows)}", flush=True)
    finally:
        h1.remove(), h2.remove(), h3.remove()


def build(model, label, a_id, b_id, rows_fit, rows_test, rng):
    pos = [r for r in rows_fit if r[label]]
    neg = [r for r in rows_fit if not r[label]]
    v = fit_readout(torch.stack([r["a"].float() for r in pos]), torch.stack([r["a"].float() for r in neg]), lam=LAM)
    sep_fit = separation(v, torch.stack([r["a"].float() for r in pos]), torch.stack([r["a"].float() for r in neg]))
    tp = [r for r in rows_test if r[label]]
    tn = [r for r in rows_test if not r[label]]
    sep_test = separation(v, torch.stack([r["a"].float() for r in tp]), torch.stack([r["a"].float() for r in tn])) if tp and tn else None
    rms_y = float(np.median([r["rms_y"] for r in pos]))
    rms_h = float(np.median([r["rms_h"] for r in pos]))
    u = write_vector(model, LAYER, a_id, b_id, C, rms_y, rms_h)
    # control: a read-out fitted to as many randomly chosen tool-name positions, same write
    random_states = set(rng.sample(range(len(rows_fit)), len(pos)))
    vp = fit_readout(torch.stack([rows_fit[i]["a"].float() for i in sorted(random_states)]),
                     torch.stack([rows_fit[i]["a"].float() for i in range(len(rows_fit)) if i not in random_states]), lam=LAM)
    return dict(v=v, u=u, vp=vp, n_pos=len(pos), n_neg=len(neg), sep_fit=sep_fit, sep_test=sep_test,
                control_on_true_positives=round(float(torch.stack([r["a"].float() for r in tp]).matmul(vp).mean()), 4) if tp else None,
                rms_y=rms_y, rms_h=rms_h)


def save_adapter(name, parts):
    """Rank = number of neurons; B = [u ...], A = [v ...]; lora_alpha = rank, so the scale is exactly 1."""
    from safetensors.torch import save_file
    B = torch.stack([u for u, v in parts], 1).float()
    A = torch.stack([v for u, v in parts], 0).float()
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    base = f"base_model.model.model.language_model.layers.{LAYER}.mlp.down_proj"
    save_file({f"{base}.lora_A.weight": A.contiguous(), f"{base}.lora_B.weight": B.contiguous()}, os.path.join(d, "adapter_model.safetensors"))
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=len(parts), lora_alpha=len(parts),
               target_modules=["down_proj"], layers_to_transform=[LAYER], lora_dropout=0.0, bias="none", fan_in_fan_out=False,
               init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    return d


with Experiment("EXP-076", "synthetic-decision-neurons", category="improvement", depends_on=["EXP-044a"],
                motivation="Add new neurons, not only edit existing ones: two read-out neurons built from official traces that act at one agent decision each.",
                hypothesis="Each read-out separates its decision on held-out tasks (test split) far above its control. Real-task effects are measured in the follow-up runs.",
                config=dict(layer=LAYER, logit_shift=C, ridge=LAM, ctx=CTX, runs=list(RUNS), fit_split="dev", check_split="test")) as ex:
    rows = states()
    rng = random.Random(76)
    keep = [r for r in rows if r["reread"] or r["test"]]
    rest = [r for r in rows if not (r["reread"] or r["test"])]
    rows = keep + rng.sample(rest, min(len(rest), 600))
    if SMOKE:
        rows = rows[:12]
    ex.log("states", len(rows), "reread positives", sum(r["reread"] for r in rows), "test positives", sum(r["test"] for r in rows),
           "dev", sum(r["split"] == "dev" for r in rows), "test", sum(r["split"] == "test" for r in rows))
    tok = tokenizer()
    from sne.model import load
    model, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    dev = device()
    gather(model, tok, rows, dev)
    fit = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    out = {}
    parts, control = [], []
    for label, a, b in (("reread", "read", "edit"), ("test", "run", "submit")):
        if not any(r[label] for r in fit):
            ex.log(label, "no positives in the fitting split")
            continue
        res = build(model, label, NAMES[a], NAMES[b], fit, test, rng)
        out[label] = {k: v for k, v in res.items() if k not in ("v", "u", "vp")}
        out[label]["base_gap_at_positives"] = round(float(np.median([r[f"gap_{a}_{b}" if f"gap_{a}_{b}" in r else ("gap_read_edit" if a == "read" else "gap_run_submit")] for r in fit if r[label]])), 3)
        ex.log(label, out[label])
        parts.append((res["u"], res["v"]))
        control.append((res["u"], res["vp"]))
        save_adapter(f"neuron_{label}", [(res["u"], res["v"])])
        save_adapter(f"control_{label}", [(res["u"], res["vp"])])
        torch.save(dict(v=res["v"], u=res["u"], vp=res["vp"]), os.path.join(OUT, f"neuron_{label}.pt")) if os.path.isdir(OUT) else None
    if len(parts) == 2:
        save_adapter("neurons_both", parts)
        save_adapter("control_both", control)
    ex.result(neurons=out, headline="; ".join(f"{k}: held-out read-out on positives {v['sep_test']['pos_mean'] if v['sep_test'] else 'n/a'}, "
                                             f"negatives p99 {v['sep_test']['neg_p99'] if v['sep_test'] else 'n/a'}, control on positives {v['control_on_true_positives']}"
                                             for k, v in out.items()))

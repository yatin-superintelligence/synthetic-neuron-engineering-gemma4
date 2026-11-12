"""EXP-106: do the transplants carry the donor's knowledge? Per-token accuracy on reference-fix tokens of held-out states, no generation.

EXP-102 and EXP-102b chose donor neurons for raising the reference token over E2B's choice where E2B was wrong on 160 fit states
(teacher-forced), and predicted gains of 0.36 (E4B) and 0.64 logits (31B) at layer 32 alone. The bench, which generates whole answers,
is too noisy to see such a change (D-099, D-102) and does not predict the agent (D-104). This run measures the quantity the transplant
was built to move, on states it never saw: for each held-out state of EXP-092's bench (dev and test, 257 states) with the reference
fix written in, E2B reads it in bf16 with each adapter merged into its weights, and at every reference-edit token (old_string and
new_string of the correct edit) the reference token's rank-1 accuracy and log-probability are kept. Also on 80 of the fit states,
to see how much of any gain is fit to the states the neurons were chosen on. Reference points from EXP-102/102b on fit states:
E2B 90.2%, E4B 92.4%, 31B 96.0% of reference-edit tokens.
Arms: unedited; E4B transplant (state map, strength 1) and its random control at strength 1; 31B transplant (logit map, strength 1)
and 31B random neurons through the logit map (strength 4, the only saved control); removal of the replaced E2B neurons (each donor).
Paired tests by token (McNemar on rank-1 correctness) and by state (sign test on mean log-probability), each arm against unedited.
"""
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

HERE = os.path.dirname(__file__)
_src = open(os.path.join(HERE, "exp092_patch_neurons.py"), encoding="utf-8").read()
P = {"__file__": os.path.join(HERE, "exp092_patch_neurons.py"), "__name__": "exp092"}
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)
STR = re.compile(r'(?:old_string|new_string):<\|"\|>(.*?)<\|"\|>', re.S)
A = os.path.join(ROOT, "artifacts_local")
ARMS = {"e4b_st_a1": f"{A}/EXP-102/st_a1", "e4b_st_rnd_a1": f"{A}/EXP-102/st_rnd_a1", "e4b_abl": f"{A}/EXP-102/abl",
        "31b_lg_a1": f"{A}/EXP-102b/lg_a1", "31b_lg_rnd_a4": f"{A}/EXP-102b/lg_rnd_a4", "31b_abl": f"{A}/EXP-102b/abl"}


def gold_message(st):
    return {"role": "assistant", "content": "", "reasoning": "This matches the failing test. I will make this one-line fix.",
            "tool_calls": [{"type": "function", "function": {"name": "edit_file", "arguments": st["gold"]}}]}


def mcnemar(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def deltas(path):
    from safetensors.torch import load_file
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    out = {}
    for k, v in sd.items():
        if k.endswith(".lora_A.weight"):
            base = k[: -len(".lora_A.weight")]
            L = int(base.split(".layers.")[1].split(".")[0])
            mod = base.rsplit(".", 1)[1]
            out[(L, mod)] = sd[base + ".lora_B.weight"].float() @ v.float()  # PEFT scaling alpha / r = 1 for every adapter here
    return out


with Experiment("EXP-106", "transplant-token-transfer", category="localization", depends_on=["EXP-102", "EXP-102b"],
                motivation="Whether the transplanted neurons move what they were chosen to move, the reference-fix tokens, on held-out states.",
                hypothesis="The 31B and E4B transplants raise E2B's rank-1 accuracy on reference-edit tokens of held-out states above the unedited "
                           "model and above their random controls.",
                config=dict(arms=sorted(ARMS), split="dev+test (257) and 80 fit states")) as ex:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from sne.model import layers, load
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:40]
    states, _ = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    ids = sorted({s["iid"] for s in states})
    random.Random(92).shuffle(ids)
    n = len(ids)
    split = {i: ("fit" if k < n // 2 else "held") for k, i in enumerate(ids)}
    held = [s for s in states if split[s["iid"]] == "held"]
    fit = [s for s in states if split[s["iid"]] == "fit"]
    random.Random(106).shuffle(fit)
    use = [("held", s) for s in held] + [("fit", s) for s in fit[: 8 if SMOKE else 80]]
    data = []
    for part, s in use:
        pre = P["G"]["render"](tok, s["prefix"], True)
        full = P["G"]["render"](tok, s["prefix"] + [gold_message(s)], False)
        if not full.startswith(pre):
            continue
        ans = full[len(pre):]
        p_ids = tok(pre, add_special_tokens=False)["input_ids"]
        enc = tok(ans, add_special_tokens=False, return_offsets_mapping=True)
        a_ids, offs = enc["input_ids"], enc["offset_mapping"]
        spans = [m.span(1) for m in STR.finditer(ans)]
        tgt = [j for j, (a, b) in enumerate(offs) if b > a and any(x <= a and b <= y for x, y in spans)]
        if not tgt or len(p_ids) + len(a_ids) > 14000:
            continue
        data.append(dict(part=part, iid=s["iid"], ids=p_ids + a_ids, pos=[len(p_ids) + j - 1 for j in tgt], gold=[a_ids[j] for j in tgt]))
    ex.log("states", len(data), "| held", sum(d["part"] == "held" for d in data), "| reference-edit tokens", sum(len(d["pos"]) for d in data))
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False)
    Ls = layers(model)
    tm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    E = model.get_output_embeddings().weight
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    st = {}
    hk = tm.norm.register_forward_hook(lambda m, a, o: st.__setitem__("hf", o[0]))

    def evaluate(tag):
        rows = []
        with torch.no_grad():
            for d in data:
                model(input_ids=torch.tensor([d["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
                z = st["hf"][torch.tensor(d["pos"], device="cuda")].float() @ E.float().T
                if cap:
                    z = cap * torch.tanh(z / cap)
                lp = torch.log_softmax(z, -1)
                g = torch.tensor(d["gold"], device="cuda")
                rows.append(dict(part=d["part"], iid=d["iid"], right=(z.argmax(-1) == g).tolist(), lp=lp[torch.arange(len(g)), g].tolist()))
        ex.log(tag, {p: round(sum(sum(r["right"]) for r in rows if r["part"] == p) / max(1, sum(len(r["right"]) for r in rows if r["part"] == p)), 4)
                     for p in ("held", "fit")})
        return rows
    res = {"base": evaluate("base")}
    for arm, path in ARMS.items():
        if not os.path.exists(path):
            ex.log("missing adapter", arm, path)
            continue
        D = deltas(path)
        orig = {}
        for (L, mod), dW in D.items():  # merge in float, keep the original bf16 tensor to restore it exactly
            w = getattr(Ls[L].mlp, mod).weight
            orig[(L, mod)] = w.data.clone()
            w.data = (w.data.float() + dW.to(w.device)).to(w.dtype)
        res[arm] = evaluate(arm)
        for (L, mod), w0 in orig.items():
            getattr(Ls[L].mlp, mod).weight.data = w0
        del orig
    hk.remove()
    ex.save_json("rows.json", res)
    summ = {}
    for arm, rows in res.items():
        for part in ("held", "fit"):
            R = [r for r in rows if r["part"] == part]
            B = [r for r in res["base"] if r["part"] == part]
            tot = sum(len(r["right"]) for r in R)
            row = dict(tokens=tot, accuracy=round(sum(sum(r["right"]) for r in R) / max(1, tot), 4),
                       mean_logprob=round(sum(sum(r["lp"]) for r in R) / max(1, tot), 4))
            if arm != "base":
                x = sum(1 for r, b in zip(R, B) for i, j in zip(r["right"], b["right"]) if i and not j)
                y = sum(1 for r, b in zip(R, B) for i, j in zip(r["right"], b["right"]) if j and not i)
                up = sum(1 for r, b in zip(R, B) if sum(r["lp"]) > sum(b["lp"]))
                dn = sum(1 for r, b in zip(R, B) if sum(r["lp"]) < sum(b["lp"]))
                row.update(tokens_gained=x, tokens_lost=y, p_tokens=round(mcnemar(x, y), 4), states_logprob_up=up, states_logprob_down=dn,
                           p_states=round(mcnemar(up, dn), 4))
            summ[f"{arm}/{part}"] = row
            ex.log(arm, part, row)
    ex.result(summary=summ, headline="held reference-edit token accuracy: " + ", ".join(f"{a} {summ[a + '/held']['accuracy']}" for a in res))

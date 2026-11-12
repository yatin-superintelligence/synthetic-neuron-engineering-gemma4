"""EXP-077: transplant neurons from E4B into E2B, chosen and mapped on real agent traces, and built as adapters for real-task tests.

EXP-080 and EXP-081 designed a transplant on a code-versus-prose corpus. This version uses what the agent actually reads and writes:
states from the official E2B traces, teacher-forced through both models on identical tokens (the tokenizers are the same file).
Steps, with no gradient training:
  1. Layer match by linear CKA of residual streams (20 states, every layer pair).
  2. For the 6 recipient layers where E2B's tool-call syntax group concentrates (layers 14 to 19, EXP-063 showed this group carries
     exact copying), fit ridge maps on dev-split states: x_R -> x_D (MLP inputs) and r_D -> r_R (residuals); held-out R^2 on
     test-split states.
  3. Donor neurons: E4B neurons tuned to the tool-call segment of a response against the thought segment (Cohen's d), top 64 per
     matched donor layer. Read fidelity: correlation, on held-out tokens, between the mapped neuron in E2B and the donor neuron in
     E4B. Controls: 64 random donor neurons through the same maps; the true map with tokens shuffled between models.
  4. Install into E2B's 64 least-used MLP slots per layer (lowest mean |activation| on the fitting tokens), neurons with fidelity
     above 0.3 only. Arms: tx_call64_k1.0, tx_call64_k0.5, tx_rand64_k1.0 (random donor neurons, same maps and slots),
     recycle_only64 (the same slots emptied, nothing installed). Adapters are built for the official stack (PEFT, rank 64 per module,
     padded so lora_alpha equals the rank and the scale is 1 in any loader).
The real-task test runs afterwards through tools/colab_batch.py. Precision: maps and adapters come from the bf16 reference models; the
real-task test is on the official W4A16 model.
"""
import gc
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, ROOT, SMOKE, loops as L  # noqa: E402
from sne.model import empty_cache, load  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.transplant import capture_all, linear_cka, map_neurons, neuron_acts, ridge  # noqa: E402

REC_LAYERS = [14, 15, 16, 17, 18, 19]
TOPN, FID_MIN, CTX, MAXTOK = 64, 0.3, 1500, 160
RUNS = {r: os.path.join(ROOT, "artifacts_local", "kaggle", r, "results", "traces") for r in ("exp029", "exp037", "exp-044a")}
SQ = '<|"|>'
OUT = os.path.join(ROOT, "artifacts_local", "EXP-077")


def call_text(c):
    args = c.get("arguments") or {}
    return "<|tool_call>call:" + c.get("function_name", "") + "{" + ",".join(f"{k}:{SQ}{v}{SQ}" for k, v in args.items()) + "}<tool_call|>"


def make_pairs(tok, n_dev, n_test, seed=77):
    """Two pairs per state: (context, thought) and (context + thought, tool call). Context is the last CTX tokens of the prompt."""
    split = L.split_map()
    states = []
    for run, d in RUNS.items():
        for f in sorted(os.listdir(d)):
            if not f.startswith("trace_"):
                continue
            t = json.load(open(os.path.join(d, f), encoding="utf-8"))
            iid = f[6:-5]
            for i, s in L.agent_steps(t):
                if s.get("tool_calls") and (s.get("message") or "").strip():
                    states.append((split.get(iid, "dev"), os.path.join(d, f), i))
    rng = random.Random(seed)
    rng.shuffle(states)
    out = {"dev": [], "test": []}
    cache = {}
    for sp, f, i in states:
        if len(out[sp]) >= (n_dev if sp == "dev" else n_test):
            continue
        t = cache.setdefault(f, json.load(open(f, encoding="utf-8")))
        s = t["steps"][i]
        ctx = L.render(tok, dict(messages=L.messages_before(t, i)))
        ids = tok(ctx, add_special_tokens=False)["input_ids"][-CTX:]
        ctx = tok.decode(ids)
        thought = s.get("message") or ""
        out[sp].append(dict(thought=(ctx, thought), call=(ctx + thought + "<channel|>", call_text(s["tool_calls"][0]))))
    return out


def pairs_of(rows):
    """Thought pairs first, then call pairs, so each capture splits into the two classes by position."""
    return [r["thought"] for r in rows] + [r["call"] for r in rows]


def n_tokens(tok, rows, key):
    return sum(min(MAXTOK, len(tok(r[key][1] + "<turn|>", add_special_tokens=False)["input_ids"])) for r in rows)


with Experiment("EXP-077", "agent-transplant-e4b-to-e2b", category="transplant", depends_on=["EXP-063", "EXP-071"],
                motivation="E4B handles the agent states that trap E2B. Transplant E4B neurons tuned to the tool-call segment into E2B's least-used slots, mapped on real agent tokens.",
                hypothesis="Ridge maps reach held-out R^2 above 0.5 at matched depths; selected donor neurons read with median fidelity above 0.5, far above shuffled tokens. Real-task effects are measured in the follow-up runs.",
                config=dict(rec_layers=REC_LAYERS, top_n=TOPN, fidelity_min=FID_MIN, ctx=CTX, max_tokens=MAXTOK, runs=list(RUNS))) as ex:
    from sne.runner import tokenizer
    tok = tokenizer()
    P = make_pairs(tok, 12 if SMOKE else 120, 6 if SMOKE else 50)
    A, B = P["dev"], P["test"]
    ntA, ntB = n_tokens(tok, A, "thought"), n_tokens(tok, B, "thought")
    ex.log("states: dev", len(A), "test", len(B), "| thought tokens dev", ntA, "test", ntB)
    # ---- recipient
    mR, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    LR = mR.model.language_model.layers
    nR = len(LR)
    cka_R = capture_all(mR, tok, pairs_of(A[:20]), list(range(nR)), max_tokens_per_seq=MAXTOK, what=("r",))
    fitR = capture_all(mR, tok, pairs_of(A), REC_LAYERS, max_tokens_per_seq=MAXTOK)
    holdR = capture_all(mR, tok, pairs_of(B), REC_LAYERS, max_tokens_per_seq=MAXTOK)
    WR = {l: dict(g=LR[l].mlp.gate_proj.weight.detach().float().cpu(), u=LR[l].mlp.up_proj.weight.detach().float().cpu(),
                  d=LR[l].mlp.down_proj.weight.detach().float().cpu(), wpost=LR[l].post_feedforward_layernorm.weight.detach().float().cpu())
          for l in REC_LAYERS}
    del mR
    gc.collect()
    empty_cache()
    # ---- donor
    mD, tokD = load(os.path.join(MODELS, "gemma-4-E4B-it-qat-unquantized"))
    assert tokD.get_vocab() == tok.get_vocab(), "tokenizers differ"
    LD = mD.model.language_model.layers
    nD = len(LD)
    cka_D = capture_all(mD, tok, pairs_of(A[:20]), list(range(nD)), max_tokens_per_seq=MAXTOK, what=("r",))
    cka = np.array([[linear_cka(cka_R[i]["r"], cka_D[j]["r"]) for j in range(nD)] for i in range(nR)])
    np.save(os.path.join(ex.dir, "cka.npy"), cka)
    # one donor layer per recipient layer, in increasing depth, maximising summed CKA (run 1 took the argmax per layer, which mapped
    # all six recipient layers onto two donor layers and would have installed the same donor neurons up to four times)
    best = {}
    def dp(k, lo):
        if k == len(REC_LAYERS):
            return 0.0, []
        if (k, lo) in best:
            return best[(k, lo)]
        out = (-1e9, [])
        for j in range(lo, nD - (len(REC_LAYERS) - k - 1)):
            v, rest = dp(k + 1, j + 1)
            if cka[REC_LAYERS[k], j] + v > out[0]:
                out = (float(cka[REC_LAYERS[k], j] + v), [j] + rest)
        best[(k, lo)] = out
        return out
    match = dict(zip(REC_LAYERS, dp(0, 0)[1]))
    ex.log("argmax donor layers (not used)", {l: int(cka[l].argmax()) for l in REC_LAYERS})
    ex.log("matched donor layers", match)
    don = sorted(set(match.values()))
    fitD = capture_all(mD, tok, pairs_of(A), don, max_tokens_per_seq=MAXTOK)
    holdD = capture_all(mD, tok, pairs_of(B), don, max_tokens_per_seq=MAXTOK)
    WD = {m: dict(g=LD[m].mlp.gate_proj.weight.detach().float().cpu(), u=LD[m].mlp.up_proj.weight.detach().float().cpu(),
                  d=LD[m].mlp.down_proj.weight.detach().float().cpu(), wpost=LD[m].post_feedforward_layernorm.weight.detach().float().cpu())
          for m in don}
    del mD
    gc.collect()
    empty_cache()
    rng = np.random.default_rng(77)
    results, mapped = {}, {}
    for l in REC_LAYERS:
        m = match[l]
        Wx, r2x_fit = ridge(fitR[l]["x"], fitD[m]["x"])
        Wr, r2r_fit = ridge(fitD[m]["r"], fitR[l]["r"])
        hx = holdR[l]["x"].float() @ Wx
        r2x = 1 - float((hx - holdD[m]["x"].float()).pow(2).sum() / (holdD[m]["x"].float() - holdD[m]["x"].float().mean(0)).pow(2).sum())
        hr = holdD[m]["r"].float() @ Wr
        r2r = 1 - float((hr - holdR[l]["r"].float()).pow(2).sum() / (holdR[l]["r"].float() - holdR[l]["r"].float().mean(0)).pow(2).sum())
        aD = neuron_acts(fitD[m]["x"], WD[m]["g"], WD[m]["u"])
        a_th, a_call = aD[:ntA], aD[ntA:]
        dsel = (a_call.mean(0) - a_th.mean(0)) / (0.5 * (a_call.var(0) + a_th.var(0))).sqrt().clamp_min(1e-6)
        top = torch.argsort(dsel, descending=True)[:TOPN]
        rnd = torch.tensor(rng.choice(WD[m]["g"].shape[0], TOPN, replace=False))
        rmsD = float(fitD[m]["y"].float().pow(2).mean().sqrt())
        rmsR = float(fitR[l]["y"].float().pow(2).mean().sqrt())

        def fidelity(sel, shuffle=False):
            gR, uR, dR = map_neurons(WD[m]["g"][sel], WD[m]["u"][sel], WD[m]["d"][:, sel], Wx, Wr, WD[m]["wpost"], WR[l]["wpost"], rmsD, rmsR)
            xr = holdR[l]["x"]
            if shuffle:
                xr = xr[torch.randperm(xr.shape[0], generator=torch.Generator().manual_seed(0))]
            aR = neuron_acts(xr, gR, uR)
            aDh = neuron_acts(holdD[m]["x"], WD[m]["g"][sel], WD[m]["u"][sel])
            aR, aDh = aR - aR.mean(0), aDh - aDh.mean(0)
            return (aR * aDh).sum(0) / (aR.norm(dim=0) * aDh.norm(dim=0)).clamp_min(1e-9), (gR, uR, dR)
        r_top, neu = fidelity(top)
        r_rnd, neu_rnd = fidelity(rnd)
        r_shuf, _ = fidelity(top, shuffle=True)
        usage = neuron_acts(fitR[l]["x"], WR[l]["g"], WR[l]["u"]).abs().mean(0)
        low = torch.argsort(usage)[:TOPN].tolist()
        mapped[l] = dict(donor_layer=m, top=top.tolist(), g=neu[0], u=neu[1], d=neu[2], fid=r_top.tolist(),
                         rand=rnd.tolist(), rg=neu_rnd[0], ru=neu_rnd[1], rd=neu_rnd[2], rfid=r_rnd.tolist(), slots=low)
        results[l] = dict(donor_layer=m, cka=round(float(cka[l, m]), 4), r2_x_fit=r2x_fit, r2_x_hold=round(r2x, 4), r2_r_fit=r2r_fit, r2_r_hold=round(r2r, 4),
                          donor_d_median=round(float(dsel[top].median()), 3), fid_top_median=round(float(r_top.median()), 4),
                          fid_rand_median=round(float(r_rnd.median()), 4), fid_shuffled_median=round(float(r_shuf.median()), 4),
                          n_top_fid_above_min=int((r_top > FID_MIN).sum()), slot_usage_max=round(float(usage[low].max()), 5),
                          usage_median=round(float(usage.median()), 5))
        ex.log("layer", l, results[l])
    ex.save_json("alignment.json", results)
    os.makedirs(OUT, exist_ok=True)
    torch.save(mapped, os.path.join(OUT, "mapped_neurons.pt"))

    # ---- adapters: each installed neuron overwrites one slot in gate, up and down (rank 1 each); modules padded to rank TOPN
    from safetensors.torch import save_file

    def adapter(name, which, kappa, recycle_only=False):
        tensors, used = {}, {}
        for l, m in mapped.items():
            sel = [i for i, f in enumerate(m["fid"] if which == "top" else m["rfid"]) if recycle_only or f > FID_MIN][:TOPN]
            slots = m["slots"][:len(sel)]
            used[l] = len(sel)
            W = WR[l]
            for mod in ("gate_proj", "up_proj", "down_proj"):
                rank = TOPN
                if mod == "down_proj":
                    A = torch.zeros(rank, W["d"].shape[1])
                    B = torch.zeros(W["d"].shape[0], rank)
                    for k, (s, i) in enumerate(zip(slots, sel)):
                        A[k, s] = 1.0
                        new = torch.zeros(W["d"].shape[0]) if recycle_only else (m["d"] if which == "top" else m["rd"])[:, i] * kappa
                        B[:, k] = new - W["d"][:, s]
                else:
                    key = "g" if mod == "gate_proj" else "u"
                    Wm = W[key]
                    A = torch.zeros(rank, Wm.shape[1])
                    B = torch.zeros(Wm.shape[0], rank)
                    if recycle_only:
                        continue  # emptying a slot needs only its down column at zero
                    for k, (s, i) in enumerate(zip(slots, sel)):
                        newrow = (m[key] if which == "top" else m["r" + key])[i]
                        A[k] = newrow - Wm[s]
                        B[s, k] = 1.0
                base = f"base_model.model.model.language_model.layers.{l}.mlp.{mod}"
                tensors[f"{base}.lora_A.weight"] = A.to(torch.float16).contiguous()
                tensors[f"{base}.lora_B.weight"] = B.to(torch.float16).contiguous()
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
        mods = ["down_proj"] if recycle_only else ["gate_proj", "up_proj", "down_proj"]
        json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=TOPN, lora_alpha=TOPN,
                       target_modules=mods, layers_to_transform=REC_LAYERS, lora_dropout=0.0, bias="none", fan_in_fan_out=False,
                       init_lora_weights=True, inference_mode=True, use_rslora=False),
                  open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
        return used

    built = {"tx_call64_k1.0": adapter("tx_call64_k1.0", "top", 1.0), "tx_call64_k0.5": adapter("tx_call64_k0.5", "top", 0.5),
             "tx_rand64_k1.0": adapter("tx_rand64_k1.0", "rand", 1.0), "recycle_only64": adapter("recycle_only64", "top", 0.0, recycle_only=True)}
    ex.save_json("adapters.json", built)
    ex.result(alignment=results, adapters=built,
              headline=f"median held-out R2(x) {np.median([v['r2_x_hold'] for v in results.values()]):.3f}; read fidelity of selected donor neurons "
                       f"{np.median([v['fid_top_median'] for v in results.values()]):.3f}, random {np.median([v['fid_rand_median'] for v in results.values()]):.3f}, "
                       f"shuffled {np.median([v['fid_shuffled_median'] for v in results.values()]):.3f}; installed per layer {built['tx_call64_k1.0']}")

"""EXP-109: the 31B transplant done with total-effect selection, judged on reference-fix tokens of unseen tasks.

What is known (D-109, D-112): 896 E4B neurons translated into E2B through the state map and chosen by their direct effect on the
logits (their write through E2B's final norm and unembedding) raise E2B's accuracy on reference-edit tokens of held-out states by
about 1.1 points (p = 2e-6), and random neurons do nothing. The 31B knows far more at this step (96.0% against 90.2% teacher-forced),
but its only token-tested transplant used the vocabulary map at late layers and gave nothing. Two things were never tried:
  1. the 31B through the state map (the map that worked for E4B), tested on tokens;
  2. choosing neurons by their total effect: a write added at layer L reaches the logits mostly through the layers after L, which the
     direct-logit score ignores (it was about 0 at layers 8 to 20 for both donors). Here, for each fit state, E2B is run in bf16 with
     gradients and the summed margin M (reference token over E2B's own top token, at every reference-edit position where E2B is wrong)
     is differentiated with respect to each chosen layer's output h_L at every position. A translated donor neuron j adds a_j(p) v_j
     at every position p, so its first-order total effect is sum_p a_j(p) (dM/dh_L(p) . v_j), summed over fit states.
Method otherwise as EXP-102 (same 160 fit states, layers 8 to 32 in steps of 4, maps, filter, placement, rank 128, strength 1), with
31B (QAT, bf16) as donor on a G4 (D-050: G4 only for 31B). Arms: st_dla (direct-logit choice), st_tot (total-effect choice), st_rnd
(random 31B neurons, same filter); reference: E4B's 7-layer transplant (EXP-102 st_a1). Measure: EXP-108's, on the same 160
held-out states (rank-1 accuracy of the reference token, exact McNemar against the unedited model).
"""
import json
import math
import os
import random
import re
import runpy
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DONOR", "31B")
EXPID = "EXP-109" if os.environ["DONOR"] == "31B" else "EXP-109e"  # EXP-109e: the same selection test with E4B as donor (fits an L4)
X = runpy.run_path(os.path.join(HERE, "exp102_neuron_transplant.py"), run_name="x102lib")
G = X["fit_stage"].__globals__  # runpy returns a copy; the functions read this dict (EXP-108's lesson)
G.update(L2=[8, 12, 16, 20, 24, 28, 32], LATE=[], EXP=EXPID, FIT_OUT=f"/content/{EXPID}_fit.json", LAB_DIR="/content/x109_adapters",
         ARMS=[("st_dla", "st", 1.0), ("st_tot", "st_tot", 1.0), ("st_rnd", "st_rnd", 1.0)])
torch = G["torch"]
Fn = G["Fn"]
P = G["P"]
ROOT = G["ROOT"]
SMOKE = G["SMOKE"]
OUT = os.path.join(ROOT, "artifacts_local", EXPID)
N_EVAL = 24 if SMOKE else 160
STR = re.compile(r'(?:old_string|new_string):<\|"\|>(.*?)<\|"\|>', re.S)
K = G["K"]


def mcnemar(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def total_effect_scores(fits, data_ids, C):
    """First-order total effect of every translated donor neuron on the summed margin, through all of E2B's later layers."""
    from huggingface_hub import snapshot_download
    from sne.model import layers, load
    model, _ = load(snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat"), ple_on_cpu=False)
    for p in model.parameters():
        p.requires_grad_(False)
    Ls = layers(model)
    tm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    E = model.get_output_embeddings().weight
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    LAY = sorted(fits)
    first = LAY[0]
    st = {}

    def keep_x(m, a, o, L):
        st[("x", L)] = o.detach()  # 1 x T x 1536

    def keep_h(m, a, o, L):
        h = o[0] if isinstance(o, tuple) else o
        if L == first:  # the graph starts here: earlier layers need no gradient
            h = h.detach().requires_grad_(True)
            st[("h", L)] = h
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        h.retain_grad()
        st[("h", L)] = h
        return None
    hooks = [tm.norm.register_forward_hook(lambda m, a, o: st.__setitem__("hf", o))]
    for L in LAY:
        hooks.append(Ls[L].pre_feedforward_layernorm.register_forward_hook(lambda m, a, o, L=L: keep_x(m, a, o, L)))
        hooks.append(Ls[L].register_forward_hook(lambda m, a, o, L=L: keep_h(m, a, o, L)))
    Wd = {L: (C["Id"]["W"][C["LD"][L]]["G"], C["Id"]["W"][C["LD"][L]]["U"]) for L in LAY}
    T = {L: (Wd[L][0] @ fits[L]["P"].T, Wd[L][1] @ fits[L]["P"].T, fits[L]["sel"]["st"][1]) for L in LAY}  # Gt, Ut, V (1536 x n_ff)
    T = {L: tuple(t.to("cuda") for t in v) for L, v in T.items()}
    score = {L: torch.zeros(T[L][0].shape[0], device="cuda", dtype=torch.float64) for L in LAY}
    t2 = C["t2"]
    off = 0
    used = 0
    for k, d in enumerate(data_ids):
        n = len(d["ans_pos"])
        top = t2[off: off + n]
        off += n
        gold = torch.tensor(d["gold"])
        wrong = (top != gold).nonzero().flatten()
        if len(wrong) == 0:
            continue
        pos = torch.tensor(d["ans_pos"])[wrong].cuda()
        model(input_ids=torch.tensor([d["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
        hf = st["hf"][0][pos].float()
        z = hf @ E.float().T
        if cap:
            z = cap * torch.tanh(z / cap)
        M = (z[torch.arange(len(pos)), gold[wrong].cuda()] - z[torch.arange(len(pos)), top[wrong].cuda()]).sum()
        M.backward()
        with torch.no_grad():
            for L in LAY:
                g = st[("h", L)].grad[0].float()  # T x 1536
                Gt, Ut, V = T[L]
                x = st[("x", L)][0].float()
                for s in range(0, x.shape[0], 1024):  # chunks keep T x n_ff small
                    a = Fn.gelu(x[s:s + 1024] @ Gt.T, approximate="tanh") * (x[s:s + 1024] @ Ut.T)
                    score[L] += (a * (g[s:s + 1024] @ V)).sum(0).double()
        st.clear()
        used += 1
        if k % 40 == 0:
            print(f"[tot] state {k}/{len(data_ids)}", flush=True)
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return {L: score[L].float().cpu() for L in LAY}, used


def fit_stage():
    """EXP-102's fit stage, plus the total-effect selection, written as arms."""
    import gc
    from transformers import AutoTokenizer
    log = lambda *a: print(time.strftime("[%H:%M:%S]"), *a, flush=True)  # noqa: E731
    ex = type("X", (), {"log": staticmethod(log)})()
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:40]
    states, _ = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    data_ids = G["make_data"](states, tok)
    cache = f"/content/{G['EXP']}_capture.pt"
    # EXP-102's fit_stage records both models (saving the capture cache), fits and writes the direct-logit and random arms; the
    # total-effect arm is added below from the same cache and fits
    G["fit_stage"]()
    C = torch.load(cache)
    rep, fits = G["analyse"](C, data_ids, log)
    scores, used = total_effect_scores(fits, data_ids, C)
    log("total-effect scores from", used, "fit states with an error")
    for L, f in fits.items():
        sc = scores[L].clone()
        top_dla = f["sel"]["st"][0]
        info = rep["layers"][L]
        sc_f = sc.clone()
        sc_f[f["corr2"] < 0.25] = -float("inf")  # the same filter as EXP-102's choice
        top = torch.argsort(sc_f, descending=True)
        top = top[sc_f[top] > 0][:K]
        f["sel"]["st_tot"] = (top, f["sel"]["st"][1])
        info["tot_selected"] = int(len(top))
        info["tot_score_sum_selected"] = round(float(sc[top].sum()), 4)
        info["tot_score_sum_dla_choice"] = round(float(sc[top_dla].sum()), 4)
        info["overlap_tot_dla"] = int(len(set(top.tolist()) & set(top_dla.tolist())))
        log(f"layer {L}", {k: info[k] for k in ("tot_selected", "tot_score_sum_selected", "tot_score_sum_dla_choice", "overlap_tot_dla")})
    arms = G["write_arms"](fits, C["I2"], C["Id"], C["LD"], G["LAB_DIR"])
    log("adapters", sorted(arms))
    json.dump(dict(rep=rep, arms=arms), open(G["FIT_OUT"], "w"), default=str)


if __name__ == "__main__" and os.environ.get("EXP109_STAGE") == "fit":
    fit_stage()
elif __name__ == "__main__":
    from sne.registry import Experiment
    with Experiment(EXPID, f"transplant-{os.environ['DONOR'].lower()}-total-effect", category="improvement", depends_on=["EXP-102b", "EXP-108"],
                    motivation="The 31B knows far more at the fix step than E4B, but its only token-tested transplant used the vocabulary map. "
                               "The state map worked for E4B, and the direct-logit choice is blind to what mid-layer neurons do downstream.",
                    hypothesis="31B neurons through the state map, chosen by total effect, raise held reference-edit token accuracy above the "
                               "unedited model, above random 31B neurons and above the direct-logit choice.",
                    config=dict(e2b_layers=G["L2"], arms=[a[0] for a in G["ARMS"]], eval_states=N_EVAL, neurons_per_layer=K, donor=G["DONOR_REPO"])) as ex:
        import shutil
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        from sne.model import layers, load
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        flog = f"/content/{EXPID}_fit.log"
        with open(flog, "w") as fl:
            fit = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, EXP109_STAGE="fit"), stdout=fl, stderr=subprocess.STDOUT)
        ex.log("fit stage exit", fit.returncode, "in", round(time.time() - t0), "s; log tail:", open(flog).read()[-5000:])
        F = json.load(open(G["FIT_OUT"]))
        ex.save_json("transplant.json", F["rep"])
        arms = F["arms"]
        ref = os.path.join(ROOT, "artifacts_local", "EXP-102", "st_a1")
        if os.path.exists(ref):
            arms["e4b_st7_a1"] = ref
        log = ex.log
        tasks, _ = P["load_tasks"]()
        if SMOKE:
            tasks = tasks[:40]
        states, _ = P["build_states"](tasks, ex)
        tok = AutoTokenizer.from_pretrained(P["official_model"]())
        ids = sorted({s["iid"] for s in states})
        random.Random(92).shuffle(ids)
        held_ids = set(ids[len(ids) // 2:])
        held = [s for s in states if s["iid"] in held_ids]
        random.Random(108).shuffle(held)  # EXP-108's held sample
        data = []
        for s in held:
            pre = P["G"]["render"](tok, s["prefix"], True)
            full = P["G"]["render"](tok, s["prefix"] + [G["gold_message"](s)], False)
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
            data.append(dict(iid=s["iid"], ids=p_ids + a_ids, pos=[len(p_ids) + j - 1 for j in tgt], gold=[a_ids[j] for j in tgt]))
            if len(data) >= N_EVAL:
                break
        log("eval states", len(data), "| reference-edit tokens", sum(len(d["pos"]) for d in data))
        model, _ = load(snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat"), ple_on_cpu=False)
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
                    rows.append(dict(iid=d["iid"], right=(z.argmax(-1) == g).tolist(), lp=lp[torch.arange(len(g)), g].tolist()))
            log(tag, "held accuracy", round(sum(sum(r["right"]) for r in rows) / max(1, sum(len(r["right"]) for r in rows)), 4))
            return rows
        res = {"base": evaluate("base")}
        for arm, path in arms.items():
            D = {}
            from safetensors.torch import load_file
            sd = load_file(os.path.join(path, "adapter_model.safetensors"))
            for k_, v in sd.items():
                if k_.endswith(".lora_A.weight"):
                    base = k_[: -len(".lora_A.weight")]
                    D[(int(base.split(".layers.")[1].split(".")[0]), base.rsplit(".", 1)[1])] = sd[base + ".lora_B.weight"].float() @ v.float()
            orig = {}
            for (L, mod), dW in D.items():
                w = getattr(Ls[L].mlp, mod).weight
                orig[(L, mod)] = w.data.clone()
                w.data = (w.data.float() + dW.to(w.device)).to(w.dtype)
            res[arm] = evaluate(arm)
            for (L, mod), w0 in orig.items():
                getattr(Ls[L].mlp, mod).weight.data = w0
            del orig, D
            if not arm.startswith("e4b"):
                shutil.copytree(path, os.path.join(OUT, arm), dirs_exist_ok=True)
        hk.remove()
        ex.save_json("rows.json", res)
        summ = {}
        B = res["base"]
        for arm, R in res.items():
            tot = sum(len(r["right"]) for r in R)
            row = dict(tokens=tot, accuracy=round(sum(sum(r["right"]) for r in R) / max(1, tot), 4), mean_logprob=round(sum(sum(r["lp"]) for r in R) / max(1, tot), 4))
            if arm != "base":
                x = sum(1 for r, b in zip(R, B) for i, j in zip(r["right"], b["right"]) if i and not j)
                y = sum(1 for r, b in zip(R, B) for i, j in zip(r["right"], b["right"]) if j and not i)
                row.update(tokens_gained=x, tokens_lost=y, p_tokens=mcnemar(x, y))
            summ[arm] = row
            log(arm, row)
        ex.result(summary=summ, layers=F["rep"]["layers"], headline="31B transplant (state map), held reference-edit token accuracy: " +
                  ", ".join(f"{a} {summ[a]['accuracy']}" for a in summ))

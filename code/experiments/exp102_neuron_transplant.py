"""EXP-102: neuron transplant. Neurons of a larger Gemma 4 (the donor, E4B by default) translated into E2B, in closed form, no training.

The author's instruction (2026-11-09): transplant neurons, and find the right way to do it. E2B is not nested inside E4B (hidden size 1536
against 2560; their weights are unrelated), so a donor neuron cannot be copied as it is. It is translated:
  record: E2B and the donor read the same token ids (one tokenizer) - fit states of EXP-092's patch bench with the reference fix
          written in - and at 7 matched depths each model's MLP input (the pre-feedforward norm's output) and layer output are kept,
          at the reference-edit tokens, the other answer tokens and 96 random prompt tokens per state;
  input:  a ridge map P from E2B's MLP input to the donor's, so a donor neuron's gate and up rows g, u become g P^T, u P^T in E2B
          (per neuron, the squared correlation of its translated activation with its true one on held-out states is reported);
  output: the donor neuron's write (its down column through the donor's post-feedforward norm and layer scalar) translated either
          (state map) by a ridge map R from the donor's layer output to E2B's, or (logit map) so that it moves E2B's logits over the
          shared 262,144-token vocabulary as it moves the donor's (least squares over E2B's unembedding); then turned into an E2B
          down column through E2B's own post-feedforward norm and layer scalar;
  place:  each transplanted neuron replaces one of E2B's 128 weakest neurons at that layer (smallest mean |activation| times write
          norm), so the edit is an exact LoRA on gate_proj, up_proj and down_proj of rank 128 (the competition's limit);
  choose: the 128 donor neurons per layer whose translated write most raises the reference token over E2B's own choice where E2B is
          wrong (first-order, through E2B's final norm and unembedding), among those whose translated activation correlates with
          the true one (squared correlation at least 0.25).
Arms (fast lab, D-082: one vLLM server, the unedited arm in the same server, 96 quick states):
  st_a1, st_a4: state map, 7 layers, write strength 1 and 4;  lg_a1, lg_a4: logit map, the 4 late layers, strength 1 and 4;
  controls: st_rnd_a4, lg_rnd_a4 - random donor neurons (same filter, same layers, same strength) transplanted the same way;
  abl - the same E2B neurons removed with nothing put in (the cost of making room).
"""
import json
import math
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

HERE = os.path.dirname(__file__)
UNIT = bool(os.environ.get("EXP102_UNIT"))
if not UNIT:
    from sne import ROOT, SMOKE  # noqa: E402
    from sne.registry import Experiment  # noqa: E402
    _src = open(os.path.join(HERE, "exp092_patch_neurons.py"), encoding="utf-8").read()
    P = {"__file__": os.path.join(HERE, "exp092_patch_neurons.py"), "__name__": "exp092"}
    exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)
else:
    ROOT, SMOKE = ".", True
DONOR = os.environ.get("DONOR", "E4B")
DONOR_REPO = {"E4B": "google/gemma-4-E4B-it-qat-q4_0-unquantized", "31B": "google/gemma-4-31B-it-qat-q4_0-unquantized"}[DONOR]
EXP = "EXP-102" if DONOR == "E4B" else "EXP-102b"
OUT = os.path.join(ROOT, "artifacts_local", EXP)
L2 = [8, 12, 16, 20, 24, 28, 32]
LATE = [20, 24, 28, 32]
K = 128
N_STATES = 24 if SMOKE else 160
N_PROMPT = 96
STR = re.compile(r'(?:old_string|new_string):<\|"\|>(.*?)<\|"\|>', re.S)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = [("st_a1", "st", 1.0), ("st_a4", "st", 4.0), ("lg_a1", "lg", 1.0), ("lg_a4", "lg", 4.0), ("st_rnd_a4", "st_rnd", 4.0),
        ("lg_rnd_a4", "lg_rnd", 4.0), ("abl", "abl", 1.0)]


def gold_message(st):
    return {"role": "assistant", "content": "", "reasoning": "This matches the failing test. I will make this one-line fix.",
            "tool_calls": [{"type": "function", "function": {"name": "edit_file", "arguments": st["gold"]}}]}


def inv_scale(w):
    """Regularised elementwise inverse of a norm scale (directions a norm nearly removes cannot be written)."""
    e = (0.05 * w.abs().median()) ** 2
    return w / (w * w + e)


def ridge(X, Y, lam_rel=1e-2):
    """argmin_B |X B - Y|^2 + lam |B|^2, lam relative to the mean eigenvalue of X^T X (float64)."""
    X, Y = X.double().to(DEV), Y.double().to(DEV)
    G = X.T @ X
    lam = lam_rel * float(torch.diagonal(G).mean())
    return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV, dtype=G.dtype), X.T @ Y).float().cpu()


def r2(pred, true):
    pred, true = pred.double(), true.double()
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean(0)) ** 2).sum())


def act(X, G, U):
    X, G, U = X.to(DEV), G.to(DEV), U.to(DEV)
    return Fn.gelu(X @ G.T, approximate="tanh") * (X @ U.T)


def text_of(model):
    return model.model.language_model if hasattr(model.model, "language_model") else model.model


def capture(model, Ls, lay, data_ids):
    """Per state: MLP input and layer output at the chosen layers and positions; final-norm rms and argmax at reference-edit positions."""
    tm = text_of(model)
    E = model.get_output_embeddings().weight
    st = {}
    hooks = [tm.norm.register_forward_hook(lambda m, a, o: st.__setitem__("hf", o[0])),
             tm.norm.register_forward_pre_hook(lambda m, a: st.__setitem__("hpre", a[0][0]))]
    for L in lay:
        hooks.append(Ls[L].pre_feedforward_layernorm.register_forward_hook(
            lambda m, a, o, L=L: st.__setitem__(("x", L), o[0][st["pos"]].to(torch.bfloat16).cpu())))
        hooks.append(Ls[L].register_forward_hook(
            lambda m, a, o, L=L: st.__setitem__(("h", L), (o[0] if isinstance(o, tuple) else o)[0][st["pos"]].to(torch.bfloat16).cpu())))
    rec = {L: dict(x=[], h=[]) for L in lay}
    tops, rmsf = [], []
    with torch.no_grad():
        for k, d in enumerate(data_ids):
            st["pos"] = torch.tensor(d["pos"], device=DEV)
            model(input_ids=torch.tensor([d["ids"]], device=DEV), use_cache=False, logits_to_keep=1)
            for L in lay:
                rec[L]["x"].append(st[("x", L)])
                rec[L]["h"].append(st[("h", L)])
            ap = torch.tensor(d["ans_pos"], device=DEV)
            tops.append((st["hf"][ap].float() @ E.float().T).argmax(-1).cpu())
            rmsf.append(st["hpre"][ap].float().pow(2).mean(-1).sqrt().cpu())
            if k % 40 == 0:
                print(f"[tp] capture {k}/{len(data_ids)}", flush=True)
    for h in hooks:
        h.remove()
    return {L: dict(x=torch.cat(v["x"]), h=torch.cat(v["h"])) for L, v in rec.items()}, torch.cat(tops), torch.cat(rmsf)


def model_info(model, Ls, lay):
    W = {L: dict(G=Ls[L].mlp.gate_proj.weight.float().cpu(), U=Ls[L].mlp.up_proj.weight.float().cpu(), D=Ls[L].mlp.down_proj.weight.float().cpu(),
                 wp=Ls[L].post_feedforward_layernorm.weight.float().cpu(), s=float(Ls[L].layer_scalar.float())) for L in lay}
    return dict(W=W, s=[float(l.layer_scalar.float()) for l in Ls], wf=text_of(model).norm.weight.float().cpu(),
                E=model.get_output_embeddings().weight.float().cpu(), n=len(Ls))


@torch.no_grad()  # no gradients here: graphs kept by stored tensors filled the GPU in EXP-109e
def fit_layer(L, Ld, X2, H2, Xd, Hd, w2, wd, s2_all, sd_all, rows, Mlog, rmsratio_final, late):
    """Translation maps, neuron quality, E2B neurons to replace, and per-arm donor selections at one layer pair.
    rows: is_test (per record row), ans_rows (rows of reference-edit positions), wr (rows where E2B is wrong), gw (margin directions
    there, d margin / d final-norm input), wf2, wfd (final norm scales)."""
    tr, te = ~rows["is_test"], rows["is_test"]
    Pm = ridge(X2[tr], Xd[tr])
    hm2, hmd = H2[tr].mean(0), Hd[tr].mean(0)
    Rm = ridge(Hd[tr] - hmd, H2[tr] - hm2)
    Gt, Ut = wd["G"] @ Pm.T, wd["U"] @ Pm.T
    at, ad = act(X2[te], Gt, Ut), act(Xd[te], wd["G"], wd["U"])
    atc, adc = at - at.mean(0), ad - ad.mean(0)
    corr2 = ((atc * adc).sum(0) ** 2 / ((atc ** 2).sum(0) * (adc ** 2).sum(0) + 1e-12)).cpu()
    del at, ad, atc, adc
    ans = rows["ans_rows"]
    yd = float((act(Xd[ans], wd["G"], wd["U"]) @ wd["D"].T.to(DEV)).pow(2).mean(-1).sqrt().median())
    a2 = act(X2, w2["G"], w2["U"])
    y2 = float((a2[ans.to(DEV)] @ w2["D"].T.to(DEV)).pow(2).mean(-1).sqrt().median())
    contrib = a2.abs().mean(0).cpu() * (w2["D"] * w2["wp"][:, None]).norm(dim=0)
    J = torch.argsort(contrib)[:K].tolist()
    del a2
    wd_out = wd["s"] * (wd["D"] * wd["wp"][:, None]) / yd  # donor write per unit activation, donor layer-output coordinates
    cum2 = float(torch.tensor(s2_all[L + 1:] + [1.0]).prod())
    cumd = float(torch.tensor(sd_all[Ld + 1:] + [1.0]).prod())
    maps = {"st": Rm.T @ wd_out}
    if L in late:
        maps["lg"] = inv_scale(rows["wf2"])[:, None] * (Mlog @ (rows["wfd"][:, None] * wd_out)) * (cumd / cum2) * rmsratio_final
    aw = act(X2[rows["wr"]], Gt, Ut).cpu()
    ok = corr2 >= 0.25
    info = dict(donor_layer=Ld, P_r2_heldout=round(r2((X2[te] @ Pm), Xd[te]), 4), R_r2_heldout=round(r2((Hd[te] - hmd) @ Rm, H2[te] - hm2), 4),
                donor_neurons=int(len(corr2)), corr2_median=round(float(corr2.median()), 4), corr2_ge_025=int(ok.sum()), rms_y_e2b=round(y2, 4),
                rms_y_donor=round(yd, 4), replaced_e2b_contrib_share=round(float(contrib[J].sum() / contrib.sum()), 5))
    sel = {}
    for tag, V in maps.items():
        eff = aw * (cum2 * (rows["gw"] @ V))  # first-order margin change, per wrong row and donor neuron, strength 1
        score = eff.mean(0)
        score[~ok] = -float("inf")
        top = torch.argsort(score, descending=True)[:K]
        top = top[score[top] > 0]
        pool = torch.nonzero(ok).flatten().tolist()
        rnd = torch.tensor(random.Random(1000 + L).sample(pool, min(K, len(pool))), dtype=torch.long)
        info[f"{tag}_selected"] = int(len(top))
        info[f"{tag}_predicted_margin_gain_a1"] = round(float(eff[:, top].sum(1).mean()), 4)
        info[f"{tag}_rnd_predicted_margin_gain_a1"] = round(float(eff[:, rnd].sum(1).mean()), 4)
        info[f"{tag}_selected_corr2_median"] = round(float(corr2[top].median()), 4) if len(top) else None
        info[f"{tag}_write_norm_ratio_selected"] = round(float(V[:, top].norm(dim=0).mean() / V.norm(dim=0).mean()), 4) if len(top) else None
        sel[tag] = (top, V)
        sel[tag + "_rnd"] = (rnd, V)
    return dict(P=Pm, J=J, y2=y2, sel=sel, info=info, corr2=corr2.detach())


def build_mods(fits, w2s, wds, which, alpha):
    """which: 'abl', or a selection tag ('st', 'lg', 'st_rnd', 'lg_rnd'). Returns save_replacement's input."""
    mods = {}
    for L, f in fits.items():
        w2 = w2s[L]
        J = f["J"]
        nff = w2["G"].shape[0]
        if which == "abl":
            mods[L] = dict(J=J, nff=nff, G=None, D=torch.zeros(w2["D"].shape[0], len(J)), D0=w2["D"][:, J])
            continue
        if which not in f["sel"]:
            continue
        idx, V = f["sel"][which]
        if len(idx) == 0:
            continue
        Jk = J[: len(idx)]
        Gt, Ut = wds[L]["G"][idx] @ f["P"].T, wds[L]["U"][idx] @ f["P"].T
        dcol = (V[:, idx] / w2["s"]) * inv_scale(w2["wp"])[:, None] * f["y2"] * alpha
        mods[L] = dict(J=Jk, nff=nff, G=Gt, U=Ut, G0=w2["G"][Jk], U0=w2["U"][Jk], D=dcol, D0=w2["D"][:, Jk])
    return mods


def _pad_rows(A):
    return torch.cat([A, torch.zeros(K - A.shape[0], A.shape[1])]) if A.shape[0] < K else A


def _pad_cols(B):
    return torch.cat([B, torch.zeros(B.shape[0], K - B.shape[1])], 1) if B.shape[1] < K else B


def save_replacement(path, mods):
    """Exact PEFT LoRA (alpha / r = 1). gate/up: delta = B A, B one-hot columns e_J, A = new - old rows. down: A one-hot rows, B = new - old columns."""
    from safetensors.torch import save_file
    sd, targets = {}, set()
    for L, m in mods.items():
        J = torch.tensor(m["J"], dtype=torch.long)
        oc = torch.zeros(m["nff"], K)
        oc[J, torch.arange(len(J))] = 1.0
        base = f"base_model.model.model.language_model.layers.{L}.mlp"
        if m.get("G") is not None:
            sd[f"{base}.gate_proj.lora_A.weight"] = _pad_rows(m["G"] - m["G0"]).to(torch.bfloat16).contiguous()
            sd[f"{base}.gate_proj.lora_B.weight"] = oc.to(torch.bfloat16).contiguous()
            sd[f"{base}.up_proj.lora_A.weight"] = _pad_rows(m["U"] - m["U0"]).to(torch.bfloat16).contiguous()
            sd[f"{base}.up_proj.lora_B.weight"] = oc.to(torch.bfloat16).contiguous()
            targets |= {"gate_proj", "up_proj"}
        sd[f"{base}.down_proj.lora_A.weight"] = oc.T.contiguous().to(torch.bfloat16)
        sd[f"{base}.down_proj.lora_B.weight"] = _pad_cols(m["D"] - m["D0"]).to(torch.bfloat16).contiguous()
        targets.add("down_proj")
    os.makedirs(path, exist_ok=True)
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=K, lora_alpha=K,
                   target_modules=sorted(targets), layers_to_transform=sorted(mods), lora_dropout=0.0, bias="none", fan_in_fan_out=False,
                   init_lora_weights=True, inference_mode=True, use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


def mcnemar(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def make_data(states, tok):
    ids = sorted({s["iid"] for s in states})
    random.Random(92).shuffle(ids)
    fit = set(ids[: len(ids) // 2])
    fit_states = [s for s in states if s["iid"] in fit]
    random.Random(102).shuffle(fit_states)
    rng = random.Random(102)
    data_ids = []
    for s in fit_states:
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
        n = len(p_ids)
        ans_pos = [n + j - 1 for j in tgt]  # the position that predicts each reference-edit token
        tg = set(tgt)
        other = [n + j - 1 for j in range(len(a_ids)) if j not in tg]
        prm = rng.sample(range(1, n - 1), min(N_PROMPT, n - 2))
        pos = sorted(set(ans_pos + other + prm))
        where = {p: i for i, p in enumerate(pos)}
        data_ids.append(dict(iid=s["iid"], ids=p_ids + a_ids, pos=pos, ans_pos=ans_pos, gold=[a_ids[j] for j in tgt], ans_idx=[where[p] for p in ans_pos]))
        if len(data_ids) >= N_STATES:
            break
    return data_ids


@torch.no_grad()  # no gradients here: graphs kept by stored tensors filled the GPU in EXP-109e
def analyse(C, data_ids, ex_log):
    I2, Id, LD = C["I2"], C["Id"], C["LD"]
    gold_all = torch.cat([torch.tensor(d["gold"]) for d in data_ids])
    wrong2 = C["t2"] != gold_all
    acc = dict(e2b=round(float((~wrong2).float().mean()), 4), donor=round(float((C["td"] == gold_all).float().mean()), 4),
               e2b_wrong_donor_right=int((wrong2 & (C["td"] == gold_all)).sum()), e2b_right_donor_wrong=int((~wrong2 & (C["td"] != gold_all)).sum()),
               positions=len(gold_all), same_tokenizer=bool(C["same_tok"]))
    ex_log("teacher-forced accuracy on reference-edit tokens:", acc)
    gdir = (I2["E"][gold_all] - I2["E"][C["t2"]]) * I2["wf"][None, :] / C["rms2"][:, None]
    ans_rows, off = [], 0
    for d in data_ids:
        ans_rows += [off + i for i in d["ans_idx"]]
        off += len(d["pos"])
    ans_rows = torch.tensor(ans_rows, dtype=torch.long)
    state_of_row = torch.cat([torch.full((len(d["pos"]),), k) for k, d in enumerate(data_ids)])
    is_test = torch.tensor([k % 5 == 0 for k in range(len(data_ids))])[state_of_row]
    E2c, Edc = I2["E"].to(DEV), Id["E"].to(DEV)
    KK = (E2c.T @ E2c).double()
    Mlog = torch.linalg.solve(KK + 1e-3 * float(torch.diagonal(KK).mean()) * torch.eye(KK.shape[0], device=DEV, dtype=KK.dtype),
                              (E2c.T @ Edc).double()).float().cpu()
    del E2c, Edc, KK
    if DEV == "cuda":
        torch.cuda.empty_cache()
    rows = dict(is_test=is_test, ans_rows=ans_rows, wr=ans_rows[wrong2], gw=gdir[wrong2], wf2=I2["wf"], wfd=Id["wf"])
    rmsratio_final = float((C["rms2"] / C["rmsd"]).median())
    rep = dict(accuracy=acc, donor_layers=LD, rms_ratio_final=round(rmsratio_final, 4), layers={})
    fits = {}
    for L in L2:
        fits[L] = fit_layer(L, LD[L], C["rec2"][L]["x"].float(), C["rec2"][L]["h"].float(), C["recd"][LD[L]]["x"].float(), C["recd"][LD[L]]["h"].float(),
                            I2["W"][L], Id["W"][LD[L]], I2["s"], Id["s"], rows, Mlog, rmsratio_final, LATE)
        rep["layers"][L] = fits[L]["info"]
        ex_log(f"layer {L} <- donor {LD[L]}", fits[L]["info"])
        if DEV == "cuda":
            torch.cuda.empty_cache()
    return rep, fits


def write_arms(fits, I2, Id, LD, lab):
    wds = {L: Id["W"][LD[L]] for L in L2}
    arms = {}
    for name, which, alpha in ARMS:
        use = {L: f for L, f in fits.items() if which in ("abl", "st", "st_rnd") or L in LATE}
        mods = build_mods(use, I2["W"], wds, which, alpha)
        if mods:
            arms[name] = save_replacement(os.path.join(lab, name), mods)
    return arms


FIT_OUT = f"/content/{EXP}_fit.json"
LAB_DIR = "/content/fastlab/adapters/x102"


def fit_stage():
    """Recording, fitting and adapter writing, in a child process: when it exits, all of its GPU memory is free for the lab's vLLM
    (2026-11-09: run in the parent, 4.3 GB stayed allocated and vLLM could not start)."""
    import gc
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from sne.model import layers, load
    log = lambda *a: print(time.strftime("[%H:%M:%S]"), *a, flush=True)  # noqa: E731
    ex = type("X", (), {"log": staticmethod(log)})()
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:40]
    states, _ = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    data_ids = make_data(states, tok)
    log("states", len(data_ids), "| positions", sum(len(d["pos"]) for d in data_ids), "| reference-edit positions", sum(len(d["ans_pos"]) for d in data_ids))
    cache = f"/content/{EXP}_capture.pt"
    if os.path.exists(cache):
        C = torch.load(cache)
    else:
        hf2 = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
        m2, _ = load(hf2, ple_on_cpu=False)
        I2 = model_info(m2, layers(m2), L2)
        rec2, t2, rms2 = capture(m2, layers(m2), L2, data_ids)
        del m2
        gc.collect()
        torch.cuda.empty_cache()
        hfd = snapshot_download(DONOR_REPO, local_dir=f"/content/hf_{DONOR.lower()}")
        if DONOR == "31B":  # 62 GB in bf16: load straight onto the large GPU, text decoder only
            from transformers import AutoModelForCausalLM
            tokd = AutoTokenizer.from_pretrained(hfd)
            md = AutoModelForCausalLM.from_pretrained(hfd, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True).eval()
            for nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
                if hasattr(md.model, nm) and getattr(md.model, nm) is not None:
                    setattr(md.model, nm, None)
            torch.cuda.empty_cache()
        else:
            md, tokd = load(hfd, ple_on_cpu=True)
        probe = data_ids[0]["ids"][:4000]
        same_tok = tokd.convert_ids_to_tokens(probe) == tok.convert_ids_to_tokens(probe)
        nd = len(layers(md))
        LD = {L: round(L * (nd - 1) / 34) for L in L2}
        Id = model_info(md, layers(md), [LD[L] for L in L2])
        recd, td, rmsd = capture(md, layers(md), [LD[L] for L in L2], data_ids)
        del md
        gc.collect()
        torch.cuda.empty_cache()
        C = dict(I2=I2, Id=Id, rec2=rec2, recd=recd, t2=t2, td=td, rms2=rms2, rmsd=rmsd, LD=LD, same_tok=same_tok)
        torch.save(C, cache)
    rep, fits = analyse(C, data_ids, log)
    arms = write_arms(fits, C["I2"], C["Id"], C["LD"], LAB_DIR)
    log("adapters", sorted(arms))
    json.dump(dict(rep=rep, arms=arms), open(FIT_OUT, "w"), default=str)


if __name__ == "__main__" and not UNIT and os.environ.get("EXP102_STAGE") == "fit":
    fit_stage()
elif __name__ == "__main__" and not UNIT:
    with Experiment(EXP, f"neuron-transplant-{DONOR.lower()}", category="improvement", depends_on=["EXP-092", "EXP-101"],
                    motivation=f"The author's instruction: transplant neurons from a larger model. Gemma 4 {DONOR} neurons translated into E2B through "
                               "fitted maps between the two models' internal states, placed in E2B's weakest neurons as an exact LoRA.",
                    hypothesis="Donor neurons chosen for raising the reference fix where E2B is wrong, translated into E2B, raise correct bench answers "
                               "against the unedited model and against random donor neurons transplanted the same way.",
                    config=dict(donor=DONOR_REPO, e2b_layers=L2, late_layers=LATE, neurons_per_layer=K, states=N_STATES, prompt_positions=N_PROMPT,
                                arms=[a for a, _, _ in ARMS])) as ex:
        import shutil
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        fit = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, EXP102_STAGE="fit"))
        ex.log("fit stage exit", fit.returncode, "in", round(time.time() - t0), "s")
        F = json.load(open(FIT_OUT))
        rep, arms = F["rep"], F["arms"]
        ex.save_json("transplant.json", rep)
        os.makedirs("/content/fastlab/jobs", exist_ok=True)
        job = f"x102_{DONOR.lower()}"
        json.dump(dict(name=job, split="quick", arms=arms), open(f"/content/fastlab/jobs/{job}.json", "w"))
        t0 = time.time()
        srv = subprocess.run(["/opt/sne/bin/python", os.path.join(HERE, "fastlab_server.py")], env=dict(os.environ, FASTLAB_ONCE="1"))
        ex.log("lab exit", srv.returncode, "in", round(time.time() - t0), "s")
        resf = f"/content/fastlab/results/{job}.json"
        if not os.path.exists(resf):  # keep every adapter so a rerun of the bench needs no new fit
            for a, p in arms.items():
                shutil.copytree(p, os.path.join(OUT, a), dirs_exist_ok=True)
            raise RuntimeError("the lab produced no result; all adapters copied to " + OUT)
        res = json.load(open(resf))
        ex.save_json("bench_quick.json", res)
        order, base = res["order"], res["arms"]["base"]
        summ = {}
        for arm, v in res["arms"].items():
            row = {k: sum(bool(v[i].get(k)) for i in order) for k in ("applies", "correct", "esc", "selfcopy")}
            if arm != "base":
                for k in ("applies", "correct"):
                    x = sum(1 for i in order if v[i].get(k) and not base[i].get(k))
                    y = sum(1 for i in order if base[i].get(k) and not v[i].get(k))
                    row[k + "_gain_loss_p"] = (x, y, round(mcnemar(x, y), 3))
            summ[arm] = row
            ex.log(arm, row)
        best = max((a for a in summ if a != "base"), key=lambda a: summ[a]["correct"])
        for a in {best, "st_rnd_a4", "lg_rnd_a4", "abl"} & set(arms):
            shutil.copytree(arms[a], os.path.join(OUT, a), dirs_exist_ok=True)
        ex.result(accuracy=rep["accuracy"], layers=rep["layers"], bench=summ, best=best,
                  headline=f"{DONOR} -> E2B transplant, quick bench (96 states): correct base {summ['base']['correct']}, "
                           + ", ".join(f"{a} {summ[a]['correct']}" for a in summ if a != "base")
                           + f"; teacher-forced edit-token accuracy E2B {rep['accuracy']['e2b']}, donor {rep['accuracy']['donor']}")

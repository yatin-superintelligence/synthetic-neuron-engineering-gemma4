"""EXP-144 (D-186): a gated synthetic neuron that ends 31B's tool-call strings, rebuilt from EXP-140c's agent-level failure.

EXP-140c's linear neuron repaired the backtick-comma fault at 57 of 62 held-out states but failed in the agent (25 of 40 against 27):
(1) the model moved to a neighbouring fused ending the neuron did not cover (a quote-comma, or a plain comma, before the next argument
name), and (2) a linear rank-1 write acts wherever its read-out is not exactly zero. This neuron changes both:
  * it reads the decision, not the token: a ridge read-out z over the MLP input x (5,376 values after the pre-feed-forward norm) is
    fitted to be 1 at every state where a string's content is complete (the faults of all three kinds, 142 states, and the 441
    correct string ends) and 0 at content tokens, separators after a closing token, string openings, markdown backticks and other
    positions (tools/analysis/build_delim_bench_v2.py, five logged 31B runs);
  * it is gated: it replaces the least-used neuron of the last MLP (gate row g = k (w_z - tau b), up row h = b, down column u, where
    b . x is about 1 at every position, so the unit's activation is GELU(k (z - tau)) and stays near zero below z = tau);
  * it writes the closing token against every fused ending seen (backtick-comma, quote-comma, comma, backtick) and the tokens that
    overtake it (competitor rounds), along the minimum-norm direction e * w * w_f, at the strength with the most in-sample repairs.
Three exact rank-1 LoRAs on layer 59 (gate_proj, up_proj, down_proj). Every effect at a labelled position is computed exactly from
one recorded pass (the edit is in the last MLP, after the last attention); the run stops if the recomputation differs from the
model's own logits by more than 1 logit. Held out by task (two folds by a hash of the task id). Gates (fixed before the run):
held-out faults where the closing token becomes the most probable at 70% or more; other states with a changed most probable token
at most 1 in 1,000; content tokens where the closing token becomes the most probable at most 1 in 1,000. Generation check over whole
calls: from the start of each distinct faulty call (up to 24) and of 40 well-formed edit calls, the model regenerates the call
greedily, unedited and edited (held-out read-out); counted: well-formed calls. Control: the same unit with a random read-out of the
same norm. Outputs: artifacts_local/EXP-144/{neuron,control} (PEFT adapters), fit.json. Runs on a Colab G4 (bf16 31B).
"""
import gzip
import json
import os
import re
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.model import layers, text_model  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("EXP140_MODEL", os.path.join(ROOT, "models", "gemma-4-E2B-it-qat-unquantized") if SMOKE else "google/gemma-4-31B-it-qat-q4_0-unquantized")
BENCH = os.path.join(ROOT, "artifacts_local", "EXP-144", "delim_bench_v2.jsonl.gz")
OUT = os.path.join(ROOT, "scratch", "preflight", "EXP-144") if SMOKE else os.path.join(ROOT, "artifacts_local", "EXP-144")
Q, TC0, TC1, TR = 52, 48, 49, 50
FUSED = [8347, 827, 236764, 236929]
POS = ("ERR", "OKC")
NEG = ("CONT", "CONT_H", "CONT_BT", "TXT_BT", "TXT_BTC", "OTHER", "SEP", "OPEN")
CONTENT = ("CONT", "CONT_H", "CONT_BT")
LAMBDAS = (0.01, 1.0) if SMOKE else (0.001, 0.01, 0.1, 1.0, 10.0)
K_GATE, TAU = 10.0, 0.5
TXT_CAP, GOOD_GEN, FAULT_GEN, GEN_NEW = 3000, (2 if SMOKE else 40), (2 if SMOKE else 24), (40 if SMOKE else 700)
KEYS = re.compile(r"([a-z_]+):$")


def fold(iid):
    return zlib.crc32(iid.encode()) % 2


def well_formed(call_ids, txt):
    t = [txt(i) for i in call_ids]
    qs = [k for k, i in enumerate(call_ids) if i == Q]
    if len(qs) % 2 or not qs:
        return False
    args, prev = {}, 0
    for m in range(0, len(qs), 2):
        key = KEYS.search("".join(t[prev:qs[m]]))
        if key:
            args[key.group(1)] = "".join(t[qs[m] + 1:qs[m + 1]])
        prev = qs[m + 1] + 1
    tail = "".join(t[prev:])
    if re.search(r"[A-Za-z0-9_]", tail.replace("}", "")):
        return False  # text after the last closed string that is not a closing brace: an argument outside its string
    if "call:edit_file" in "".join(t[:12]):
        return {"filepath", "old_string", "new_string"} <= set(args) and bool(args["old_string"])
    return bool(args)


def save_lora(name, layer, i, dgate, dup, ddown, n_inter, base_name):
    from safetensors.torch import save_file
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    pre = f"base_model.model.model.language_model.layers.{layer}.mlp"
    e = torch.zeros(n_inter, 1)
    e[i, 0] = 1.0
    t = {f"{pre}.gate_proj.lora_A.weight": dgate.view(1, -1).float().contiguous(), f"{pre}.gate_proj.lora_B.weight": e.clone(),
         f"{pre}.up_proj.lora_A.weight": dup.view(1, -1).float().contiguous(), f"{pre}.up_proj.lora_B.weight": e.clone(),
         f"{pre}.down_proj.lora_A.weight": e.view(1, -1).clone(), f"{pre}.down_proj.lora_B.weight": ddown.view(-1, 1).float().contiguous()}
    save_file(t, os.path.join(d, "adapter_model.safetensors"))
    rp = {f"model.language_model.layers.{layer}.mlp.{m}": 1 for m in ("gate_proj", "up_proj", "down_proj")}
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=base_name, r=1, lora_alpha=1, rank_pattern=rp,
               alpha_pattern={k: 1 for k in rp}, target_modules=["gate_proj", "up_proj", "down_proj"], lora_dropout=0.0, bias="none",
               fan_in_fan_out=False, init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    return d


class Unit:
    """The replaced neuron i of the last MLP, applied exactly: y' = y - W_down[:, i] a_i + u a_new, a_new = act(g . x) (h . x)."""

    def __init__(self, mlp):
        self.mlp = mlp
        self.on = False
        self.h = mlp.register_forward_hook(self._hook)

    def set(self, i=None, g=None, hv=None, u=None, dev=None):
        self.on = i is not None
        if self.on:
            self.i, self.g, self.hv, self.u = i, g.float().to(dev), hv.float().to(dev), u.float().to(dev)
            self.wg = self.mlp.gate_proj.weight[i].float()
            self.wu = self.mlp.up_proj.weight[i].float()
            self.wd = self.mlp.down_proj.weight[:, i].float()

    def delta(self, x):
        xf = x.float()
        a_old = self.mlp.act_fn(xf @ self.wg) * (xf @ self.wu)
        a_new = self.mlp.act_fn(xf @ self.g) * (xf @ self.hv)
        return -a_old.unsqueeze(-1) * self.wd + a_new.unsqueeze(-1) * self.u

    def _hook(self, m, inp, out):
        if not self.on:
            return None
        return (out.float() + self.delta(inp[0])).to(out.dtype)


def capture(model, rows, layer, dev, ex):
    L = layers(model)[layer]
    st = {}
    hs = [L.pre_feedforward_layernorm.register_forward_pre_hook(lambda m, a: st.__setitem__("resid", a[0][0])),
          L.mlp.register_forward_hook(lambda m, a, o: st.update(x=a[0][0], y=o[0]) or None),
          L.mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0]))]
    keep = dict(cls=[], iid=[], row=[], pos=[], resid=[], y=[], x=[], lg=[])
    usage = None
    t0 = time.time()
    try:
        for n, r in enumerate(rows):
            P = [len(r["prompt_ids"]) + i - 1 for i, _ in r["labels"]]
            ids = r["prompt_ids"] + r["output_ids"]
            try:
                with torch.no_grad():
                    out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=torch.tensor(P, device=dev))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                ex.log("row", n, "skipped: out of memory at", len(ids), "tokens")
                continue
            ix = torch.tensor(P, device=st["y"].device)
            keep["cls"] += [c for _, c in r["labels"]]
            keep["iid"] += [r["iid"]] * len(P)
            keep["row"] += [n] * len(P)
            keep["pos"] += [i for i, _ in r["labels"]]
            keep["resid"].append(st["resid"][ix].float().cpu())
            keep["y"].append(st["y"][ix].float().cpu())
            keep["x"].append(st["x"][ix].float().cpu())
            keep["lg"].append(out.logits[0].float().cpu()[:, [Q] + FUSED])
            au = st["a"].float().abs().sum(0)
            usage = au if usage is None else usage + au
            del out
            if n % 25 == 0:
                ex.log(f"capture row {n + 1} of {len(rows)}, {len(ids)} tokens, {time.time() - t0:.0f} s so far")
    finally:
        for h in hs:
            h.remove()
    for k in ("resid", "y", "x", "lg"):
        keep[k] = torch.cat(keep[k])
    keep["usage"] = usage.cpu()
    return keep


class Head:
    def __init__(self, model, layer, dev):
        self.L = layers(model)[layer]
        assert not getattr(self.L, "enable_moe_block", False) and not getattr(self.L, "hidden_size_per_layer_input", 0) or SMOKE
        self.norm = text_model(model).norm
        self.head = model.lm_head
        self.cap = model.config.get_text_config().final_logit_softcapping
        self.dev = dev

    @torch.no_grad()
    def summary(self, cap, sel, unit=None, chunk=256):
        tops, lqs, los = [], [], []
        for s in range(0, len(sel), chunk):
            idx = torch.as_tensor(sel[s:s + chunk])
            y = cap["y"][idx].to(self.dev)
            if unit is not None:
                y = y + unit.delta(cap["x"][idx].to(self.dev))
            h = (cap["resid"][idx].to(self.dev) + self.L.post_feedforward_layernorm(y)) * self.L.layer_scalar.float()
            lg = self.head(self.norm(h).to(self.head.weight.dtype)).float()
            if self.cap:
                lg = torch.tanh(lg / self.cap) * self.cap
            tops.append(lg.argmax(-1).cpu())
            lqs.append(lg[:, Q].cpu())
            lg[:, Q] = -1e9
            los.append(lg.max(-1).values.cpu())
        return torch.cat(tops).numpy(), torch.cat(lqs).numpy(), torch.cat(los).numpy()


def ridge(X, t, w, lam, dev):
    Xs = X.to(dev, torch.float64) * w.to(dev, torch.float64).sqrt()[:, None]
    K = Xs @ Xs.T
    a = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype, device=K.device), w.to(dev, torch.float64).sqrt() * t.to(dev, torch.float64))
    return (Xs.T @ a).float().cpu()


with Experiment("EXP-144", "31b-gated-stringend-neuron", category="improvement", depends_on=["EXP-140c", "EXP-140a", "EXP-140d"],
                motivation="EXP-140c repaired one fused string ending at the token level but failed in the agent: the model moved to "
                           "another fused ending, and a linear write acts wherever its read-out is not zero. A gated unit that reads "
                           "'the string's content is complete' and holds down every fused ending addresses both.",
                hypothesis="Held out by task, the closing token becomes the most probable at 70% or more of fault states of all three "
                           "kinds with at most 1 in 1,000 other states changed, and regenerated whole calls are well formed more often "
                           "than unedited, with no well-formed call lost.",
                config=dict(model=REPO, bench="artifacts_local/EXP-144/delim_bench_v2.jsonl.gz", positives=POS, negatives=NEG, lambdas=LAMBDAS,
                            gate=dict(k=K_GATE, tau=TAU), fused=FUSED, layer="last")) as ex:
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(BENCH, "rt", encoding="utf-8")]
    if SMOKE:
        f0 = [r for r in rows if r["fault"] and fold(r["iid"]) == 0][:2]
        f1 = [r for r in rows if r["fault"] and fold(r["iid"]) == 1][:2]
        g = [r for r in rows if not r["fault"] and "edit_file" in r["calls"]]
        rows = f0 + f1 + [r for r in g if fold(r["iid"]) == 0][:2] + [r for r in g if fold(r["iid"]) == 1][:2]
        for r in rows:
            r["prompt_ids"] = r["prompt_ids"][-1000:]
    ex.log("bench rows", len(rows), "| faulty", sum(bool(r["fault"]) for r in rows), "| folds of faulty tasks",
           sorted({(r["iid"], fold(r["iid"])) for r in rows if r["fault"]})[:30])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sne.chunked_attention import install
    install()
    path = REPO if os.path.isdir(REPO) else ("/content/hf_31b" if os.path.isdir("/content/hf_31b") else None)
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO, local_dir="/content/hf_31b")
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    if torch.cuda.is_available():
        dev = fit_dev = "cuda"
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True).eval()
        for nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
            if hasattr(model.model, nm) and getattr(model.model, nm) is not None:
                setattr(model.model, nm, None)
    else:
        from sne.model import device as _device, load as _load
        model, _ = _load(path)
        dev, fit_dev = _device(), "cpu"
    layer = len(layers(model)) - 1
    L = layers(model)[layer]
    ex.log("model loaded in", round(time.time() - t0), "s | layers", layer + 1, "| device", dev)
    _txt = {}
    txt = lambda i: _txt.setdefault(i, tok.decode([i]))  # noqa: E731

    cap = capture(model, rows, layer, dev, ex)
    cls = np.array(cap["cls"])
    fo = np.array([fold(t) for t in cap["iid"]])
    err, okc = cls == "ERR", cls == "OKC"
    neg = np.isin(cls, NEG)
    content = np.isin(cls, CONTENT)
    head = Head(model, layer, dev)
    top0, lq0, lo0 = head.summary(cap, np.arange(len(cls)))
    diff = float(np.max(np.abs(lq0 - cap["lg"][:, 0].numpy())))
    out = dict(positions={c: int((cls == c).sum()) for c in sorted(set(cls.tolist()))}, exactness_max_abs_logit_difference=round(diff, 4),
               unedited=dict(ERR=int(err.sum()), ERR_top_is_close=int((top0[err] == Q).sum()), OKC_top_is_close=int((top0[okc] == Q).sum()), OKC=int(okc.sum())))
    ex.log("exactness", round(diff, 4), "| unedited", out["unedited"])
    if diff > 1.0 and not SMOKE:
        ex.save_json("fit.json", out)
        ex.result(headline=f"stopped: the recomputed logits differ from the model's by {round(diff, 3)}", adapters=[])
        raise SystemExit(1)

    # ---- read-out z on the MLP input, and the bias direction b ----
    X = cap["x"]
    rng = np.random.default_rng(144)
    use = np.ones(len(cls), bool)
    tb = np.where(cls == "TXT_BT")[0]
    if len(tb) > TXT_CAP:
        use[rng.choice(tb, len(tb) - TXT_CAP, replace=False)] = False

    def design(keep):
        idx = np.where(keep & np.isin(cls, POS + NEG))[0]
        t = torch.tensor(np.isin(cls[idx], POS).astype(np.float64))
        w = torch.ones(len(idx), dtype=torch.float64)
        for c in POS:
            m = cls[idx] == c
            if m.sum():
                w[torch.as_tensor(m)] = float((~np.isin(cls[idx], POS)).sum()) / len(POS) / m.sum()
        return idx, t, w
    bi = np.where(use)[0]
    b = ridge(X[bi], torch.ones(len(bi), dtype=torch.float64), torch.ones(len(bi), dtype=torch.float64), 1e-4, fit_dev)
    bx = (X @ b).numpy()
    out["bias_direction"] = dict(mean=round(float(bx.mean()), 4), std=round(float(bx.std()), 4), p1=round(float(np.quantile(bx, 0.01)), 4), p99=round(float(np.quantile(bx, 0.99)), 4))
    ex.log("bias direction b . x:", out["bias_direction"])
    act = L.mlp.act_fn

    def gated(z):
        return (act(torch.as_tensor(K_GATE * (z - TAU))) / act(torch.tensor(K_GATE * (1 - TAU)))).numpy()
    grid, ws = [], {}
    for f in (0, 1):
        idx, t, w = design(use & (fo != f))
        for lam in LAMBDAS:
            ws[(lam, f)] = ridge(X[idx], t, w, lam, fit_dev)
    for lam in LAMBDAS:
        z = np.full(len(cls), np.nan)
        for f in (0, 1):
            te = np.where(fo == f)[0]
            z[te] = (X[te] @ ws[(lam, f)]).numpy()
        gz = gated(z)
        grid.append(dict(lam=lam, z_ERR_mean=round(float(np.mean(z[err])), 3), gate_ERR_ge_half=round(float(np.mean(gz[err] >= 0.5)), 3),
                         gate_OKC_ge_half=round(float(np.mean(gz[okc] >= 0.5)), 3), gate_neg_p999=round(float(np.quantile(gz[neg], 0.999)), 4),
                         gate_content_p999=round(float(np.quantile(gz[content], 0.999)), 4)))
        ex.log("held-out read-out", grid[-1])
    out["grid"] = grid
    ok = [g for g in grid if g["gate_neg_p999"] < 0.05]
    best = max(ok or (grid if SMOKE else []), key=lambda g: g["gate_ERR_ge_half"], default=None)
    out["chosen"] = best
    ex.save_json("fit.json", out)
    if best is None:
        ex.result(headline="no ridge strength keeps the gate closed at negatives (99.9th percentile under 0.05); no neuron built", adapters=[])
        raise SystemExit(0)
    lam = best["lam"]
    idx, t, w = design(use)
    w_all = ridge(X[idx], t, w, lam, fit_dev)
    w_fold = {f: ws[(lam, f)] for f in (0, 1)}
    torch.manual_seed(144)
    w_rnd = torch.randn_like(w_all)
    w_rnd = w_rnd / w_rnd.norm() * w_all.norm()

    # ---- the neuron to replace, and the write ----
    wd_norm = L.mlp.down_proj.weight.float().norm(dim=0).cpu()
    score = cap["usage"] * wd_norm
    i_rep = int(torch.argmin(score))
    out["replaced_neuron"] = dict(index=i_rep, usage_score=round(float(score[i_rep]), 6), median_score=round(float(score.median()), 6))
    tm = text_model(model)
    E = tm.embed_tokens.weight
    wpf = L.post_feedforward_layernorm.weight.float()
    wfin = tm.norm.weight.float()
    g_at_one = float(act(torch.tensor(K_GATE * (1 - TAU))))
    unit = Unit(L.mlp)
    sel_err = np.where(err)[0]

    def make(wz, comp, alpha):
        e = (E[Q].float() - E.index_select(0, torch.tensor(comp, device=E.device)).float().mean(0))
        gdir = e * wpf * wfin
        u = gdir / gdir.norm() * alpha / g_at_one
        g = K_GATE * (wz.to(u.device) - TAU * b.to(u.device))
        return g, b.to(u.device), u
    comp, rounds, best_w = list(FUSED), [], None
    grid_a = np.geomspace(0.1, 20.0, 9 if SMOKE else 41)
    for rd in range(6):
        best_r = None
        for a in grid_a:
            g, hv, u = make(w_all, comp, float(a))
            unit.set(i_rep, g, hv, u, dev)
            tp, lq, lo = head.summary(cap, sel_err, unit)
            k = int((tp == Q).sum())
            if best_r is None or k > best_r[0]:
                best_r = (k, float(a), tp)
        unit.set()
        k, a, tp = best_r
        others = sorted({int(x) for x in tp if int(x) != Q} - set(comp))
        rounds.append(dict(round=rd + 1, competitors=list(comp), alpha=round(a, 4), close_is_top=k, n=int(len(tp)), new_top_tokens=[(x, txt(x)) for x in others][:12]))
        ex.log("write round", rounds[-1])
        if best_w is None or k > best_w[0]:
            best_w = (k, a, list(comp))
        if not others:
            break
        comp = comp + others[:8]
    _, alpha, comp = best_w
    out["write"] = dict(rounds=rounds, alpha=round(alpha, 4), competitors=comp)

    # ---- held-out effect, exact ----
    res = {}
    for arm, wz_of in (("neuron", lambda f: w_fold[f]), ("control", lambda f: w_rnd)):
        tops = np.zeros(len(cls), dtype=np.int64)
        for f in (0, 1):
            sel = np.where(fo == f)[0]
            g, hv, u = make(wz_of(f), comp, alpha)
            unit.set(i_rep, g, hv, u, dev)
            tp, _, _ = head.summary(cap, sel, unit)
            tops[sel] = tp
        unit.set()
        res[arm] = dict(ERR_close_top=int((tops[err] == Q).sum()), ERR_n=int(err.sum()), OKC_close_top=int((tops[okc] == Q).sum()), OKC_n=int(okc.sum()),
                        negatives_changed=int((tops[neg] != top0[neg]).sum()), negatives_n=int(neg.sum()),
                        content_to_close=int(((tops[content] == Q) & (top0[content] != Q)).sum()), content_n=int(content.sum()),
                        by_fault={kind: [int(((tops == Q) & err & m).sum()), int((err & m).sum())] for kind, m in
                                  (("backtick-comma", np.isin(np.array(cap["row"]), [n for n, r in enumerate(rows) if r["fault"] == "backtick-comma"])),
                                   ("quote-comma", np.isin(np.array(cap["row"]), [n for n, r in enumerate(rows) if r["fault"] == "quote-comma"])),
                                   ("comma", np.isin(np.array(cap["row"]), [n for n, r in enumerate(rows) if r["fault"] == "comma"])))})
        ex.log("held-out", arm, res[arm])
    out["held_out"] = res
    h = res["neuron"]
    gates = dict(faults_70_percent=h["ERR_close_top"] >= 0.7 * h["ERR_n"], negatives_1_in_1000=h["negatives_changed"] <= 0.001 * h["negatives_n"],
                 content_1_in_1000=h["content_to_close"] <= 0.001 * h["content_n"])
    out["gates"] = {k: bool(v) for k, v in gates.items()}
    ex.log("gates", out["gates"])
    ex.save_json("fit.json", out)

    # ---- generation check over whole calls ----
    stop = [TC1, TR] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    seen, faulty_rows = set(), []
    for r in rows:
        if r["fault"]:
            s0 = r["output_ids"].index(TC0)
            k = (r["iid"], tuple(r["output_ids"][s0:s0 + 40]))
            if k not in seen:
                seen.add(k)
                faulty_rows.append(r)
    good_rows = [r for r in rows if not r["fault"] and "edit_file" in r["calls"] and TC0 in r["output_ids"]][:GOOD_GEN]
    gen = dict(faulty=[], good=[])
    for kind, rs in (("faulty", faulty_rows[:FAULT_GEN]), ("good", good_rows)):
        for r in rs:
            s0 = r["output_ids"].index(TC0)
            prefix = torch.tensor([r["prompt_ids"] + r["output_ids"][:s0 + 1]], device=dev)
            row = dict(iid=r["iid"], fault=r["fault"])
            for arm in ("unedited", "edited"):
                if arm == "edited":
                    g, hv, u = make(w_fold[fold(r["iid"])], comp, alpha)
                    unit.set(i_rep, g, hv, u, dev)
                try:
                    with torch.no_grad():
                        o = model.generate(prefix, max_new_tokens=GEN_NEW, do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id or 0, use_cache=True)
                    new = [x for x in o[0, prefix.shape[1]:].tolist() if x not in (TC1, TR)]
                    row[arm] = bool(well_formed([TC0] + new, txt))
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row[arm] = None
                unit.set()
            gen[kind].append(row)
            ex.log("generation", kind, row)
    def cnt(rs, arm):
        return sum(1 for x in rs if x.get(arm))
    out["generation"] = dict(faulty_n=len(gen["faulty"]), faulty_unedited=cnt(gen["faulty"], "unedited"), faulty_edited=cnt(gen["faulty"], "edited"),
                             good_n=len(gen["good"]), good_unedited=cnt(gen["good"], "unedited"), good_edited=cnt(gen["good"], "edited"), rows=gen)
    ex.log("generation summary", {k: v for k, v in out["generation"].items() if k != "rows"})
    gen_ok = out["generation"]["faulty_edited"] > out["generation"]["faulty_unedited"] and out["generation"]["good_edited"] >= out["generation"]["good_unedited"]
    out["gates"]["generation"] = bool(gen_ok)

    # ---- adapters ----
    saved = []
    n_inter = L.mlp.down_proj.in_features
    if all(out["gates"].values()) or SMOKE:
        for name, wz in (("neuron", w_all), ("control", w_rnd)):
            g, hv, u = make(wz, comp, alpha)
            dg = g.cpu() - L.mlp.gate_proj.weight[i_rep].float().cpu()
            du = hv.cpu() - L.mlp.up_proj.weight[i_rep].float().cpu()
            dd = u.cpu() - L.mlp.down_proj.weight[:, i_rep].float().cpu()
            saved.append(save_lora(name, layer, i_rep, dg, du, dd, n_inter, "google/gemma-4-31B-it"))
    out["adapters"] = [os.path.basename(s) for s in saved]
    ex.save_json("fit.json", out)
    ex.result(fit={k: v for k, v in out.items() if k not in ("grid", "generation")}, generation={k: v for k, v in out["generation"].items() if k != "rows"},
              adapters=out["adapters"],
              headline=(f"31B gated string-end neuron ({'saved' if saved else 'gates not all passed, not saved'}): held-out, the closing token on top at "
                        f"{h['ERR_close_top']} of {h['ERR_n']} fault states (control {res['control']['ERR_close_top']}); other states changed "
                        f"{h['negatives_changed']} of {h['negatives_n']}; content to closing token {h['content_to_close']} of {h['content_n']}; whole calls "
                        f"regenerated well formed: faulty {out['generation']['faulty_edited']} of {out['generation']['faulty_n']} edited against "
                        f"{out['generation']['faulty_unedited']} unedited, good {out['generation']['good_edited']} against {out['generation']['good_unedited']}"))

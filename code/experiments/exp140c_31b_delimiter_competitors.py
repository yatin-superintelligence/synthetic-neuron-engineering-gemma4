"""EXP-140c (D-173): EXP-140b's string-delimiter neuron for Gemma 4 31B, with a write that also holds down the tokens that overtake
the closing token, and its strength chosen on a fine grid.

EXP-140b's minimum-norm write along g = e * w * w_f (e = E_close - mean(E_backtick-comma, E_backtick)) made the closing token the most
probable at 35 of 62 faulty states at its first strength, but at twice that strength at none: a rare token (id 228276, "nilvadipine")
whose embedding lines up with g under the norm weights rose faster than the closing token. The bisection never ran, because the target
margin was never reached, and the strength stayed at the first grid point. Changes, everything else as EXP-140b (read-out, exact
recomputation, held-out design, gates, generation check, control): (1) the competitor set: start from backtick-comma and backtick; at the
chosen strength add every token other than the closing one that is the most probable at a faulty state, and rebuild e = E_close - mean
over the set; up to six rounds. (2) The strength: 41 points from 0.1 to 20 (log spaced) per round; the one with the most faulty states
where the closing token is the most probable (all-task read-out, in sample), the smaller on ties. Gates unchanged (D-167): held-out
flips at 70% or more of the faulty states, at most 1 in 1,000 other states changed. The recorded test states are saved on the runtime
(/content/exp140c_cap.pt) for any later variant.
Outputs: artifacts_local/EXP-140c/delim and delim_control, fit.json.
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
BENCH = os.path.join(ROOT, "artifacts_local", "EXP-140", "delim_bench.jsonl.gz")
CACHE = "/content/exp140_acts.pt"
OUT = os.path.join(ROOT, "scratch", "preflight", "EXP-140c") if SMOKE else os.path.join(ROOT, "artifacts_local", "EXP-140c")
Q, BTC, TICK, TC0, TC1, TR = 52, 8347, 236929, 48, 49, 50
LAMBDAS = (0.01, 1.0) if SMOKE else (0.001, 0.01, 0.1, 1.0, 10.0)
NEG = ("CONT", "CONT_H", "CONT_BT", "TXT_BT", "TXT_BTC", "OTHER")
HARD = ("CONT_H", "CONT_BT", "TXT_BT", "TXT_BTC")
FOLD_A = {"fastapi_9753", "rich_3470", "rich_3454", "fastapi_11194", "fastapi_14794"}
FOLD_B = {"rich_3944", "fastapi_14258", "rich_3067", "rich_3472"}
TXT_CAP, GOOD_TEST, GEN_STATES, GEN_NEW = 3000, (2 if SMOKE else 90), (1 if SMOKE else 16), (20 if SMOKE else 300)
TARGET = 2.0
KEYS = re.compile(r"([a-z_]+):$")


def fold(iid):
    return 0 if iid in FOLD_A else 1 if iid in FOLD_B else zlib.crc32(iid.encode()) % 2


def save_adapter(name, layer, u, v, base_name):
    from safetensors.torch import save_file
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    key = f"base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
    save_file({f"{key}.lora_A.weight": v.view(1, -1).float().contiguous(), f"{key}.lora_B.weight": u.view(-1, 1).float().contiguous()},
              os.path.join(d, "adapter_model.safetensors"))
    rp = {f"model.language_model.layers.{layer}.mlp.down_proj": 1}
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=base_name, r=1, lora_alpha=1, rank_pattern=rp,
               alpha_pattern={k: 1 for k in rp}, target_modules=["down_proj"], lora_dropout=0.0, bias="none", fan_in_fan_out=False,
               init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    return d


class Edit:
    """Adds u (v . a) to the output of one down_proj while active: the LoRA's exact effect, computed in float32."""

    def __init__(self, module):
        self.u = self.v = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        if self.u is None:
            return None
        r = inp[0].float() @ self.v
        return (out.float() + r.unsqueeze(-1) * self.u).to(out.dtype)

    def set(self, u=None, v=None, dev=None):
        self.u = None if u is None else u.float().to(dev)
        self.v = None if v is None else v.float().to(dev)


def gram(X, w, dev):
    Xs = X.to(dev, torch.float64) * w.to(dev, torch.float64).sqrt()[:, None]
    return Xs, Xs @ Xs.T


def solve(Xs, K, w, t, lam):
    s = w.to(K.device, torch.float64).sqrt()
    alpha = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype, device=K.device), s * t.to(K.device, torch.float64))
    return (Xs.T @ alpha).float().cpu()


def design(cls, keep):
    cls = np.array(cls)
    sel = keep & np.isin(cls, ("ERR",) + NEG)
    idx = np.where(sel)[0]
    t = (cls[idx] == "ERR").astype(np.float64)
    w = np.ones(len(idx))
    m = t == 1
    if m.sum():
        w[m] = float((~m).sum()) / m.sum()
    return idx, torch.tensor(t), torch.tensor(w)


def well_formed(call_ids, txt):
    t = [txt(i) for i in call_ids]
    for k in range(len(t) - 1):
        if t[k].endswith(chr(96) + ",") and re.match(r"^[a-z_]{3,20}:", "".join(t[k + 1:k + 8])):
            return False
    qs = [k for k, i in enumerate(call_ids) if i == Q]
    if len(qs) % 2 or not qs:
        return False
    args, prev = {}, 0
    for m in range(0, len(qs), 2):
        key = KEYS.search("".join(t[prev:qs[m]]))
        if key:
            args[key.group(1)] = "".join(t[qs[m] + 1:qs[m + 1]])
        prev = qs[m + 1] + 1
    return {"filepath", "old_string", "new_string"} <= set(args) and bool(args["old_string"])


def capture(model, rows, layer, dev, ex, want_acts):
    """One pass per row. At every labelled position: the residual entering the last MLP branch, the MLP output y, the per-layer input
    when the model has one, the activations entering down_proj (when asked) and the model's own logits of interest."""
    L = layers(model)[layer]
    st = {}
    hs = [L.pre_feedforward_layernorm.register_forward_pre_hook(lambda m, a: st.__setitem__("resid", a[0][0])),
          L.mlp.register_forward_hook(lambda m, a, o: st.__setitem__("y", o[0])),
          L.mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0]))]
    if getattr(L, "hidden_size_per_layer_input", 0) and os.environ.get("SNE_NO_PLE_HOOK") != "1":  # E2B and E4B only; 31B has none
        hs.append(L.register_forward_pre_hook(lambda m, a, kw: st.__setitem__("pli", kw.get("per_layer_input")), with_kwargs=True))
    keep = dict(cls=[], iid=[], row=[], pos=[], resid=[], y=[], pli=[], a=[], lg=[])
    t0, skipped = time.time(), 0
    try:
        for n, r in enumerate(rows):
            P = [len(r["prompt_ids"]) + i - 1 for i, _ in r["labels"]]
            ids = r["prompt_ids"] + r["output_ids"]
            try:
                with torch.no_grad():
                    out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=torch.tensor(P, device=dev))
            except torch.OutOfMemoryError:
                skipped += 1
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
            if st.get("pli") is not None:
                keep["pli"].append(st["pli"][0][ix.to(st["pli"].device)].float().cpu())
            if want_acts:
                keep["a"].append(st["a"][ix.to(st["a"].device)].half().cpu())
            keep["lg"].append(out.logits[0].float().cpu()[:, [Q, BTC, TICK]])  # list indexing on the laptop GPU crashes its driver
            del out
            if n % 25 == 0:
                ex.log(f"capture row {n + 1} of {len(rows)}, {len(ids)} tokens, {time.time() - t0:.0f} s so far")
    finally:
        for h in hs:
            h.remove()
    for k in ("resid", "y", "pli", "a", "lg"):
        keep[k] = torch.cat(keep[k]) if keep[k] else None
    keep["skipped"] = skipped
    return keep


class Head:
    """Recomputes the logits of a position from its residual and MLP output with the model's own last-layer modules."""

    def __init__(self, model, layer, dev):
        self.L = layers(model)[layer]
        assert not getattr(self.L, "enable_moe_block", False)
        self.norm = text_model(model).norm
        self.head = model.lm_head
        self.cap = model.config.get_text_config().final_logit_softcapping
        self.dev = dev

    @torch.no_grad()
    def hidden(self, resid, y, pli=None, add=None, r=None):
        """The last layer's output for given residuals and MLP outputs (an optional edit u (v . a) added to y)."""
        yy = y.to(self.dev)
        if add is not None:
            yy = yy + r.to(self.dev).unsqueeze(-1) * add.to(self.dev)
        h = resid.to(self.dev) + self.L.post_feedforward_layernorm(yy)
        if getattr(self.L, "hidden_size_per_layer_input", 0) and pli is not None:
            r2 = h
            g = self.L.act_fn(self.L.per_layer_input_gate(h.to(self.L.per_layer_input_gate.weight.dtype))).float()
            g = self.L.per_layer_projection((g * pli.to(self.dev)).to(self.L.per_layer_projection.weight.dtype)).float()
            h = r2 + self.L.post_per_layer_input_norm(g)
        return h * self.L.layer_scalar.float()

    @torch.no_grad()
    def logits(self, resid, y, pli=None, add=None, r=None, chunk=256):
        outs = []
        for s in range(0, len(y), chunk):
            sl = slice(s, s + chunk)
            h = self.hidden(resid[sl], y[sl], None if pli is None else pli[sl], add, None if r is None else r[sl])
            lg = self.head(self.norm(h).to(self.head.weight.dtype)).float()
            if self.cap:
                lg = torch.tanh(lg / self.cap) * self.cap
            outs.append(lg)
        return torch.cat(outs)

    @torch.no_grad()
    def summary(self, cap, sel=None, add=None, r=None):
        """Per position: most probable token, logit of the closing token, best other logit."""
        idx = torch.arange(len(cap["y"])) if sel is None else torch.as_tensor(sel)
        lg = self.logits(cap["resid"][idx], cap["y"][idx], None if cap["pli"] is None else cap["pli"][idx], add, None if r is None else r[idx])
        top = lg.argmax(-1)
        lq = lg[:, Q].clone()
        lg[:, Q] = -1e9
        return top.cpu().numpy(), lq.cpu().numpy(), lg.max(-1).values.cpu().numpy(), lg[:, BTC].cpu().numpy()


with Experiment("EXP-140c", "31b-delimiter-neuron-competitor-write", category="improvement", depends_on=["EXP-140", "EXP-140b"],
                motivation="EXP-140b's minimum-norm write flipped 35 of 62 faulty states at its first strength and none at twice that, "
                           "because a rare token rose faster than the closing token; the write must hold down the tokens that overtake it.",
                hypothesis="With the same read-out, the minimum-norm write makes the closing token the most probable at 70% or more of "
                           "held-out faulty states while at most 1 in 1,000 other states change their most probable token.",
                config=dict(model=REPO, bench="artifacts_local/EXP-140/delim_bench.jsonl.gz", lambdas=LAMBDAS, write="u = alpha g / |g|, g = e * w * w_f, e = E_close - mean(E_competitors)", rounds=6, alpha_grid="41 points, 0.1 to 20",
                            target_margin_p10=TARGET, folds=dict(a=sorted(FOLD_A), b=sorted(FOLD_B)),
                            gates="held-out ERR flipped to the closing token >= 70%; held-out negatives with a changed top token <= 0.1%")) as ex:
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(BENCH, "rt", encoding="utf-8")]
    if SMOKE:
        a = [r for r in rows if r["fault"] == "ERR" and r["iid"] in FOLD_A][:2]
        b = [r for r in rows if r["fault"] == "ERR" and r["iid"] in FOLD_B][:2]
        g = [r for r in rows if not r["fault"]]
        rows = a + b + [r for r in g if fold(r["iid"]) == 0][:2] + [r for r in g if fold(r["iid"]) == 1][:2]
        for r in rows:
            r["prompt_ids"] = r["prompt_ids"][-1200:]
    ridx = {id(r): n for n, r in enumerate(rows)}
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
    ex.log("model loaded in", round(time.time() - t0), "s | layers", layer + 1, "| device", dev)
    _txt = {}
    txt = lambda i: _txt.setdefault(i, tok.decode([i]))  # noqa: E731

    # ---- the read-out: EXP-140's activations and rule ----
    faulty = [r for r in rows if r["fault"]]
    good = [r for r in rows if not r["fault"]]
    test = faulty + good[:GOOD_TEST]
    if os.path.exists(CACHE) and not SMOKE:
        c0 = torch.load(CACHE, weights_only=False)  # written by EXP-140 on this runtime (numpy arrays inside)
        X, cls_all, iid_all = c0["X"], c0["cls"], c0["iid"]
        assert len(cls_all) == sum(len(r["labels"]) for r in rows), "cache does not match the bench"
        ex.log("read-out activations from EXP-140's cache:", tuple(X.shape))
        ca = None
    else:  # one pass over every row: the read-out activations, and the test rows' recorded states for later
        ca = capture(model, rows, layer, dev, ex, True)
        X, cls_all, iid_all = ca["a"], ca["cls"], ca["iid"]
        if not SMOKE:
            torch.save(dict(X=X, cls=ca["cls"], iid=ca["iid"], row=ca["row"], pos=ca["pos"]), CACHE)
    cls_all = np.array(cls_all)
    fo_all = np.array([fold(t) for t in iid_all])
    rng = np.random.default_rng(140)
    use = np.ones(len(cls_all), bool)
    tb = np.where(cls_all == "TXT_BT")[0]
    if len(tb) > TXT_CAP:
        use[rng.choice(tb, len(tb) - TXT_CAP, replace=False)] = False
    Xf = X.float()
    err_all = cls_all == "ERR"
    grid, vs = [], {}
    for f in (0, 1):
        idx, t, w = design(cls_all, use & (fo_all != f))
        Xs, K = gram(Xf[idx], w, fit_dev)
        for lam in LAMBDAS:
            vs[(lam, f)] = solve(Xs, K, w, t, lam)
        del Xs, K
    for lam in LAMBDAS:
        r = np.full(len(cls_all), np.nan)
        for f in (0, 1):
            te = np.where(fo_all == f)[0]
            r[te] = (Xf[te] @ vs[(lam, f)]).numpy()
        hard = r[np.isin(cls_all, HARD)]
        grid.append(dict(lam=lam, pos_mean=round(float(np.mean(r[err_all])), 3), hard_p99=round(float(np.quantile(hard, 0.99)), 4)))
    ex.log("held-out read-out grid:", grid)
    ok = [g for g in grid if g["hard_p99"] < 0.1 and g["pos_mean"] >= 0.5]
    best = max(ok or (grid if SMOKE else []), key=lambda g: g["pos_mean"], default=None)
    out = dict(grid=grid, chosen=best)
    if best is None:
        ex.save_json("fit.json", out)
        ex.result(headline="no ridge strength passed the read-out rule; no neuron built", adapters=[])
        raise SystemExit(0)
    lam = best["lam"]
    idx, t, w = design(cls_all, use)
    Xs, K = gram(Xf[idx], w, fit_dev)
    v_all = solve(Xs, K, w, t, lam)
    del Xs, K
    v_fold = {f: vs[(lam, f)] for f in (0, 1)}
    torch.manual_seed(140)
    v_rnd = torch.randn_like(v_all)
    v_rnd = v_rnd / v_rnd.norm() * v_all.norm()
    del Xf

    # ---- the recorded states of the test rows (taken from the full pass when there was one) ----
    if ca is not None:
        tset = {ridx[id(r)] for r in test}
        m = torch.as_tensor(np.isin(np.array(ca["row"]), list(tset)))
        mi = np.where(m.numpy())[0]
        cap = {k: (ca[k][m] if isinstance(ca[k], torch.Tensor) else [ca[k][j] for j in mi] if isinstance(ca[k], list) else ca[k]) for k in ca}
        ex.log("test rows taken from the full pass:", len(tset), "rows,", int(m.sum()), "positions")
    else:
        cap = capture(model, test, layer, dev, ex, True)
    if not SMOKE:
        torch.save({k: cap[k] for k in ("cls", "iid", "row", "pos", "resid", "y", "a", "lg")}, "/content/exp140c_cap.pt")
    cls = np.array(cap["cls"])
    err = cls == "ERR"
    neg = np.isin(cls, NEG)
    fo = np.array([fold(t) for t in cap["iid"]])
    A = cap["a"].float()
    r_all, r_rnd = A @ v_all, A @ v_rnd
    r_held = torch.zeros(len(cls))
    for f in (0, 1):
        m = torch.as_tensor(fo == f)
        r_held[m] = A[m] @ v_fold[f]
    del A
    head = Head(model, layer, dev)
    top0, lq0, lo0, lb0 = head.summary(cap)
    own = cap["lg"].numpy()
    diff = float(np.max(np.abs(np.stack([lq0, lb0], 1) - own[:, [0, 1]])))
    out["exactness"] = dict(max_abs_logit_difference_recomputed_vs_model=round(diff, 4))
    ex.log("recomputed against the model's own logits: largest difference", round(diff, 4), "logits")
    if diff > 1.0 and not SMOKE:  # every later number rests on the recomputation
        ex.save_json("fit.json", out)
        ex.result(headline=f"stopped: the recomputed logits differ from the model's by {round(diff, 3)}", adapters=[])
        raise SystemExit(1)
    out["unedited"] = dict(ERR=int(err.sum()), ERR_top_is_backtick_comma=int((top0[err] == BTC).sum()), ERR_top_is_close=int((top0[err] == Q).sum()))

    # ---- the two writes ----
    Lm = layers(model)[layer]
    tm = text_model(model)
    E = tm.embed_tokens.weight
    e = (E[Q].float() - E[[BTC, TICK]].float().mean(0)).cpu()
    wpf = Lm.post_feedforward_layernorm.weight.float().cpu()
    wfin = tm.norm.weight.float().cpu()
    g = e * wpf * wfin
    g_hat = g / g.norm()
    rms_y = float(cap["y"][torch.as_tensor(err)].pow(2).mean(-1).sqrt().mean())
    me = torch.as_tensor(err)
    rms_h = float(head.hidden(cap["resid"][me], cap["y"][me], None if cap["pli"] is None else cap["pli"][me]).pow(2).mean(-1).sqrt().mean())
    s = float(Lm.layer_scalar.float())
    u_old_unit = rms_y * rms_h / s * (e / e.pow(2).sum()) / (wpf * wfin)
    out["direction"] = dict(norm_ratio_old_to_min_for_equal_first_order_effect=round(float((e / (wpf * wfin)).norm() * g.norm() / e.pow(2).sum()), 1))

    def measure(u, r, sel):
        tp, lq, lo, lb = head.summary(cap, sel, u, r)
        return tp, lq - lo

    sel_err = np.where(err)[0]
    old = []
    for c in (9.0, 12.0):
        tp, mg = measure(u_old_unit * c, r_all, sel_err)
        old.append(dict(c=c, close_is_top=int((tp == Q).sum()), margin_median=round(float(np.median(mg)), 2), margin_p10=round(float(np.quantile(mg, 0.1)), 2)))
    out["old_write_recomputed"] = old
    ex.log("old write, recomputed exactly (EXP-140 measured close_is_top 8 and 10 at c 9 and 12):", old)
    comp = [BTC, TICK]
    rounds, alpha, g_hat = [], None, None
    grid_a = np.geomspace(0.1, 20.0, 9 if SMOKE else 41)
    for rd in range(6):
        e_r = (E[Q].float() - E[comp].float().mean(0)).cpu()
        g_r = e_r * wpf * wfin
        gh = g_r / g_r.norm()
        best_r = None
        for a in grid_a:
            tp, mg = measure(gh * float(a), r_all, sel_err)
            k = int((tp == Q).sum())
            if best_r is None or k > best_r[0]:
                best_r = (k, float(a), tp, mg)
        k, a, tp, mg = best_r
        others = sorted({int(t) for t in tp if int(t) != Q} - set(comp))
        rounds.append(dict(round=rd + 1, competitors=list(comp), alpha=round(a, 4), close_is_top=k, n=int(len(tp)),
                           margin_median=round(float(np.median(mg)), 2), margin_p10=round(float(np.quantile(mg, 0.1)), 2),
                           new_top_tokens=[(t, txt(t)) for t in others][:12]))
        ex.log("write round", rounds[-1])
        if alpha is None or k > max(r["close_is_top"] for r in rounds[:-1]):
            alpha, g_hat, best_comp = a, gh, list(comp)
        if not others:
            break
        comp = comp + others[:8]
    u = g_hat * alpha
    tp, mg = measure(u, r_all, sel_err)
    out["write"] = dict(rounds=rounds, competitors=best_comp, alpha=round(alpha, 4), u_norm=round(float(u.norm()), 4), old_u_norm_at_c12=round(float((u_old_unit * 12).norm()), 4),
                        in_sample=dict(close_is_top=int((tp == Q).sum()), n=int(len(tp)), margin_median=round(float(np.median(mg)), 2), margin_p10=round(float(np.quantile(mg, 0.1)), 2)))
    ex.log("competitor write:", {k: v for k, v in out["write"].items() if k != "rounds"})
    ex.save_json("fit.json", out)

    # ---- held-out: every state read by the neuron fitted without its fold ----
    tpH, _, _, _ = head.summary(cap, None, u, r_held)
    tpR, _, _, _ = head.summary(cap, None, u, r_rnd)
    by = {}
    for c in sorted(set(cls.tolist())):
        m = cls == c
        by[c] = dict(n=int(m.sum()), top_changed=int((tpH[m] != top0[m]).sum()), top_is_close_before=int((top0[m] == Q).sum()),
                     top_is_close_after=int((tpH[m] == Q).sum()), control_top_changed=int((tpR[m] != top0[m]).sum()))
    flipped = int((tpH[err] == Q).sum())
    changed = int((tpH[neg] != top0[neg]).sum())
    out["held_out_edit"] = dict(by_class=by, ERR_flipped_to_close=flipped, ERR_n=int(err.sum()), negatives_changed=changed, negatives_n=int(neg.sum()),
                                control_ERR_flipped=int((tpR[err] == Q).sum()), control_negatives_changed=int((tpR[neg] != top0[neg]).sum()),
                                readout_held=dict(ERR_mean=round(float(r_held[torch.as_tensor(err)].mean()), 3),
                                                  neg_p99=round(float(torch.quantile(r_held[torch.as_tensor(neg)], 0.99)), 4)))
    g1 = flipped >= 0.7 * err.sum()
    g2 = changed <= 0.001 * neg.sum()
    out["gates"] = dict(ERR_flipped_at_least_70_percent=bool(g1), negatives_changed_at_most_1_in_1000=bool(g2))
    ex.log("held-out edit:", {k: v for k, v in out["held_out_edit"].items() if k != "by_class"}, "| gates", out["gates"])
    ex.save_json("fit.json", out)

    # ---- generation check on the model itself ----
    edit = Edit(Lm.mlp.down_proj)
    seen, states = set(), []
    for r in faulty:
        k = (r["iid"], r["call_chars"], r["fault"])
        if k not in seen:
            seen.add(k)
            states.append(r)
    gen = []
    stop = [TC1, TR] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    for r in states[:GEN_STATES]:
        i = next(p for p, c in r["labels"] if c in ("ERR", "ERR2"))
        s0 = max(k for k in range(i) if r["output_ids"][k] == TC0)
        prefix = torch.tensor([r["prompt_ids"] + r["output_ids"][:i]], device=dev)
        row = dict(iid=r["iid"], fault=r["fault"], served_token=r["output_ids"][i])
        for arm in ("unedited", "edited"):
            edit.set(*((u, v_fold[fold(r["iid"])], dev) if arm == "edited" else ()))
            try:
                with torch.no_grad():
                    gg = model.generate(prefix, max_new_tokens=GEN_NEW, do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id or 0, use_cache=True)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                row[arm] = dict(error="out of memory")
                continue
            new = gg[0, prefix.shape[1]:].tolist()
            call = r["output_ids"][s0 + 1:i] + [x for x in new if x not in (TC1, TR)]
            row[arm] = dict(first_token=new[0] if new else None, new_tokens=len(new), ended=bool(new) and new[-1] in stop, well_formed=well_formed(call, txt))
        edit.set()
        gen.append(row)
        ex.log("generation", row)
    done = [x for x in gen if "error" not in x.get("unedited", {}) and "error" not in x.get("edited", {})]
    out["generation"] = dict(states=len(gen), unedited_repeats_the_served_token=sum(x["unedited"]["first_token"] == x["served_token"] for x in done),
                             unedited_well_formed=sum(x["unedited"]["well_formed"] for x in done), edited_first_token_is_close=sum(x["edited"]["first_token"] == Q for x in done),
                             edited_well_formed=sum(x["edited"]["well_formed"] for x in done), rows=gen)
    saved = []
    if (g1 and g2) or SMOKE:
        base_name = "google/gemma-4-31B-it"
        saved = [save_adapter("delim", layer, u, v_all, base_name), save_adapter("delim_control", layer, u, v_rnd, base_name)]
    ex.save_json("fit.json", out)
    ho = out["held_out_edit"]
    ex.result(fit={k: v for k, v in out.items() if k not in ("grid", "generation")}, generation={k: v for k, v in out["generation"].items() if k != "rows"},
              adapters=[os.path.basename(x) for x in saved],
              headline=(f"31B delimiter neuron with the competitor write ({'saved' if saved else 'gates not passed, not saved'}): held-out, the closing "
                        f"token is the most probable at {ho['ERR_flipped_to_close']} of {ho['ERR_n']} faulty states (control {ho['control_ERR_flipped']}); "
                        f"{ho['negatives_changed']} of {ho['negatives_n']} other states change (control {ho['control_negatives_changed']}); greedy "
                        f"continuations well formed {out['generation']['edited_well_formed']} of {len(done)} edited against "
                        f"{out['generation']['unedited_well_formed']} unedited; write norm {out['write']['u_norm']} against {out['write']['old_u_norm_at_c12']}"))

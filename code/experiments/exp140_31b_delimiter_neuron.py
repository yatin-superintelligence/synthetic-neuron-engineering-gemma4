"""EXP-140 (D-167): a closed-form neuron that makes Gemma 4 31B end tool-call strings with the right token.

The fault. In Gemma 4's tool-call format every string argument is opened and closed by the token <|"|> (id 52). In its unedited
official runs 31B ends the new_string of some edit_file calls with the single token backtick-comma (id 8347) instead of <|"|> and a
comma. The parser never sees the string end, old_string is lost and the harness refuses the call; 31B then repeats the same call
(38 to 58 times on three tasks) until the run dies. 61 of 222 edit_file calls in EXP-074c, 49 of 155 in EXP-129a and 551 of 646 in
EXP-131a were refused this way.

The neuron. One new unit in the last layer's MLP: D' = D + u v^T, an exact rank-1 LoRA.
  v reads the neuron activations entering down_proj. Weighted ridge, closed form: about 1 at states where 31B is about to end a
    string with backtick-comma (class ERR of the bench), about 0 inside string content, at markdown backticks in thoughts and text,
    and at other positions. Variant A fits ERR alone as positives; variant B adds the states where 31B closes a string correctly.
  u writes the closing token over backtick-comma and the plain backtick (through the final norm and the unembedding), by c logits
    where the read-out is 1. c is calibrated on the model, not taken from the first-order formula.
Data: tools/analysis/build_delim_bench.py, 282 requests exactly as served in EXP-129a and EXP-131a (prompt and output token ids from
vLLM's request log), teacher-forced on the bf16 QAT checkpoint.
Held-out design: tasks are split in two folds; every reported read-out and every flip is measured with a neuron fitted on the other
fold's tasks. The saved neuron is fitted on all tasks.
Selection (fixed before the run): over variant and ridge strength, the largest held-out mean read-out at ERR with the 99th percentile
at hard negatives (content after a line end, content backticks, markdown backticks) under 0.1; the mean must be 0.5 or more.
Gates for saving the adapter: with held-out neurons, at least 70% of ERR states have <|"|> as the most probable token, and at most
1 in 1,000 negative states change their most probable token. Otherwise no adapter is saved.
Generation check: from each distinct faulty state the model continues greedily, unedited and with the held-out neuron; a call counts
as well formed when its strings pair up and it has filepath, old_string (not empty) and new_string.
Control: the same write with a random read-out of the same norm (seed 140).
Outputs (artifacts_local/EXP-140): delim and delim_control (PEFT adapters), fit.json; the gathered activations stay on the runtime. Runs on a Colab G4 (the bf16 31B is
62.6 GB).
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

from sne import ROOT, SMOKE, conditional as C  # noqa: E402
from sne.model import layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("EXP140_MODEL", os.path.join(ROOT, "models", "gemma-4-E2B-it-qat-unquantized") if SMOKE else "google/gemma-4-31B-it-qat-q4_0-unquantized")
BENCH = os.path.join(ROOT, "artifacts_local", "EXP-140", "delim_bench.jsonl.gz")
OUT = os.path.join(ROOT, "scratch", "preflight", "EXP-140") if SMOKE else os.path.join(ROOT, "artifacts_local", "EXP-140")
Q, BTC, TICK, TC0, TC1, TR = 52, 8347, 236929, 48, 49, 50
LAMBDAS = (0.01, 1.0) if SMOKE else (0.001, 0.01, 0.1, 1.0, 10.0)
NEG = ("CONT", "CONT_H", "CONT_BT", "TXT_BT", "TXT_BTC", "OTHER")
HARD = ("CONT_H", "CONT_BT", "TXT_BT", "TXT_BTC")
FOLD_A = {"fastapi_9753", "rich_3470", "rich_3454", "fastapi_11194", "fastapi_14794"}  # 26 of the 62 ERR states
FOLD_B = {"rich_3944", "fastapi_14258", "rich_3067", "rich_3472"}  # 36 of the 62 ERR states
TXT_CAP, GOOD_TEST, GEN_STATES, GEN_NEW = 3000, (2 if SMOKE else 90), (1 if SMOKE else 16), (20 if SMOKE else 300)
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


def run_rows(model, rows, layer, dev, want_acts, ex, tag):
    """One teacher-forced pass per row. Per labelled position: class, task, row index, the logits of interest and, when asked, the
    neuron activations entering the last down_proj. The state that predicts output token i sits at len(prompt) + i - 1."""
    Ls = layers(model)
    st = {}
    h1 = Ls[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0]))
    h2 = Ls[layer].mlp.register_forward_hook(lambda m, a, o: st.__setitem__("y", o[0]))
    h3 = Ls[layer].register_forward_hook(lambda m, a, o: st.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0]))
    rec = dict(cls=[], iid=[], row=[], pos=[], top=[], lq=[], lb=[], lt=[], ltop=[], lother=[], rms_y=[], rms_h=[])
    acts, t0, skipped = [], time.time(), 0
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
                ex.log(tag, "row", n, "skipped: out of memory at", len(ids), "tokens")
                continue
            lg = out.logits[0].float()
            idx = torch.tensor(P, device=lg.device)
            top = lg.argmax(-1)
            other = lg.clone()
            other[:, Q] = -1e9
            rec["cls"] += [c for _, c in r["labels"]]
            rec["iid"] += [r["iid"]] * len(P)
            rec["row"] += [n] * len(P)
            rec["pos"] += [i for i, _ in r["labels"]]
            rec["top"] += top.tolist()
            rec["lq"] += lg[:, Q].tolist()
            rec["lb"] += lg[:, BTC].tolist()
            rec["lt"] += lg[:, TICK].tolist()
            rec["ltop"] += lg.max(-1).values.tolist()
            rec["lother"] += other.max(-1).values.tolist()
            rec["rms_y"] += st["y"].float()[idx.to(st["y"].device)].pow(2).mean(-1).sqrt().tolist()
            rec["rms_h"] += st["h"].float()[idx.to(st["h"].device)].pow(2).mean(-1).sqrt().tolist()
            if want_acts:
                acts.append(st["a"][idx.to(st["a"].device)].half().cpu())
            del out, lg, other
            if n % 25 == 0:
                ex.log(tag, f"row {n + 1} of {len(rows)}, {len(ids)} tokens, {time.time() - t0:.0f} s so far")
    finally:
        h1.remove(), h2.remove(), h3.remove()
    rec = {k: (np.array(v) if k not in ("cls", "iid") else v) for k, v in rec.items()}
    rec["skipped"] = skipped
    return rec, (torch.cat(acts) if acts else None)


def gram(X, w, dev):
    Xs = X.to(dev, torch.float64) * w.to(dev, torch.float64).sqrt()[:, None]
    return Xs, Xs @ Xs.T


def solve(Xs, K, w, t, lam):
    s = w.to(K.device, torch.float64).sqrt()
    alpha = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype, device=K.device), s * t.to(K.device, torch.float64))
    return (Xs.T @ alpha).float().cpu()


def design(cls, variant, keep):
    """Targets and weights for the rows in keep: ERR is positive; variant B adds OKC; each positive class carries the weight of all
    negatives divided by the number of positive classes."""
    cls = np.array(cls)
    posc = ("ERR",) if variant == "A" else ("ERR", "OKC")
    sel = keep & np.isin(cls, posc + NEG)
    idx = np.where(sel)[0]
    t = np.isin(cls[idx], posc).astype(np.float64)
    w = np.ones(len(idx))
    nneg = max(1.0, float((t == 0).sum()))
    for c in posc:
        m = cls[idx] == c
        if m.sum():
            w[m] = nneg / len(posc) / m.sum()
    return idx, torch.tensor(t), torch.tensor(w)


def stats(r_pos, r_hard, r_neg):
    q = lambda a, p: round(float(np.quantile(a, p)), 4) if len(a) else None  # noqa: E731
    return dict(n_pos=len(r_pos), pos_mean=round(float(np.mean(r_pos)), 3) if len(r_pos) else 0.0, pos_min=q(r_pos, 0.0),
                pos_frac_ge_half=round(float(np.mean(r_pos >= 0.5)), 3) if len(r_pos) else 0.0, n_hard=len(r_hard), hard_p99=q(r_hard, 0.99),
                hard_p999=q(r_hard, 0.999), hard_max=q(r_hard, 1.0), n_neg=len(r_neg), neg_p99=q(r_neg, 0.99), neg_max=q(r_neg, 1.0))


def well_formed(call_ids, txt):
    """True when the call's strings pair up, no string ends with backtick-comma before an argument name, and filepath, old_string
    (not empty) and new_string are all present."""
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


with Experiment("EXP-140", "31b-string-delimiter-neuron", category="improvement", depends_on=["EXP-074c", "EXP-129a", "EXP-131a", "EXP-051"],
                motivation="31B ends the new_string of some edit_file calls with the token backtick-comma where the string-closing token "
                           "belongs; the call is refused and repeated until the run dies. One neuron that reads that state and writes "
                           "the closing token would repair the call at its first wrong token.",
                hypothesis="The state before the faulty token is linearly readable in the last layer on tasks the fit did not see "
                           "(held-out mean read-out 0.5 or more with hard-negative 99th percentile under 0.1), and the write makes "
                           "the closing token the most probable one there without changing other positions.",
                config=dict(model=REPO, bench="artifacts_local/EXP-140/delim_bench.jsonl.gz", lambdas=LAMBDAS, variants=("A: ERR", "B: ERR and correct closes"),
                            folds=dict(a=sorted(FOLD_A), b=sorted(FOLD_B), others="crc32 of the task id, modulo 2"),
                            selection_rule="max held-out ERR mean read-out with hard-negative p99 < 0.1 and mean >= 0.5",
                            gates="held-out ERR flipped to the closing token >= 70%; held-out negatives with a changed top token <= 0.1%",
                            tokens=dict(close=Q, backtick_comma=BTC, backtick=TICK))) as ex:
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(BENCH, "rt", encoding="utf-8")]
    if SMOKE:
        a = [r for r in rows if r["fault"] == "ERR" and r["iid"] in FOLD_A][:2]
        b = [r for r in rows if r["fault"] == "ERR" and r["iid"] in FOLD_B][:2]
        g = [r for r in rows if not r["fault"]]
        rows = a + b + [r for r in g if fold(r["iid"]) == 0][:2] + [r for r in g if fold(r["iid"]) == 1][:2]
        for r in rows:
            r["prompt_ids"] = r["prompt_ids"][-1200:]
    ex.log("bench rows", len(rows), "| faulty", sum(bool(r["fault"]) for r in rows), "| tokens", sum(len(r["prompt_ids"]) + len(r["output_ids"]) for r in rows))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sne.chunked_attention import install
    install()
    if os.path.isdir(REPO):
        path = REPO
    else:
        from huggingface_hub import snapshot_download
        t0 = time.time()
        path = snapshot_download(REPO, local_dir="/content/hf_31b")
        ex.log("model downloaded in", round(time.time() - t0), "s")
    tok = AutoTokenizer.from_pretrained(path)
    assert tok.convert_tokens_to_ids("<|" + chr(34) + "|>") == Q and tok.decode([BTC]) == chr(96) + "," and tok.decode([TICK]) == chr(96)
    t0 = time.time()
    if torch.cuda.is_available():
        dev = fit_dev = "cuda"
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True).eval()
        for nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
            if hasattr(model.model, nm) and getattr(model.model, nm) is not None:
                setattr(model.model, nm, None)
    else:  # the laptop preflight: the library's loader (its device, per-layer embeddings on the CPU); the ridge is solved on the CPU
        from sne.model import device as _device, load as _load
        model, _ = _load(path)
        dev, fit_dev = _device(), "cpu"
    layer = len(layers(model)) - 1
    ex.log("model", REPO, "loaded in", round(time.time() - t0), "s | layers", layer + 1, "| neurons entering the last down_proj",
           layers(model)[layer].mlp.down_proj.in_features, "| device", dev)
    _txt = {}
    txt = lambda i: _txt.setdefault(i, tok.decode([i]))  # noqa: E731

    # ---- pass 1: the unedited model on every row ----
    rec, X = run_rows(model, rows, layer, dev, True, ex, "pass 1")
    # the activations are large (about 0.7 GB for 31B): on the runtime they stay beside the model, outside the folder that is sent home
    torch.save(dict(X=X, cls=rec["cls"], iid=rec["iid"], row=rec["row"], pos=rec["pos"]),
               "/content/exp140_acts.pt" if os.path.isdir("/content/hf_31b") else os.path.join(OUT, "acts.pt"))
    cls = np.array(rec["cls"])
    fo = np.array([fold(t) for t in rec["iid"]])
    err = cls == "ERR"
    out = dict(rows=len(rows), rows_skipped=rec["skipped"], positions={c: int((cls == c).sum()) for c in sorted(set(rec["cls"]))}, layer=layer)
    gap = rec["lb"][err] - rec["lq"][err]
    out["unedited_at_ERR"] = dict(
        n=int(err.sum()), top_is_backtick_comma=int((rec["top"][err] == BTC).sum()), top_is_close=int((rec["top"][err] == Q).sum()),
        gap_backtick_comma_minus_close=dict(median=round(float(np.median(gap)), 2), p10=round(float(np.quantile(gap, 0.1)), 2), p90=round(float(np.quantile(gap, 0.9)), 2)),
        close_behind_top=dict(median=round(float(np.median(rec["ltop"][err] - rec["lq"][err])), 2), p90=round(float(np.quantile(rec["ltop"][err] - rec["lq"][err], 0.9)), 2)))
    okc = cls == "OKC"
    out["unedited_at_correct_closes"] = dict(n=int(okc.sum()), top_is_close=int((rec["top"][okc] == Q).sum()))
    ex.log("unedited model at ERR states:", out["unedited_at_ERR"], "| at correct closes:", out["unedited_at_correct_closes"])
    ex.save_json("fit.json", out)

    # ---- fit: held-out by task fold, then on all tasks ----
    rng = np.random.default_rng(140)
    use = np.ones(len(cls), bool)
    tb = np.where(cls == "TXT_BT")[0]
    if len(tb) > TXT_CAP:  # markdown backticks are plentiful: a sample is fitted, all are evaluated
        use[rng.choice(tb, len(tb) - TXT_CAP, replace=False)] = False
    Xf = X.float()
    grid, vs = [], {}
    for variant in ("A", "B"):
        held = {lam: np.full(len(cls), np.nan) for lam in LAMBDAS}
        for f in (0, 1):
            idx, t, w = design(cls, variant, use & (fo != f))
            if not len(idx) or t.sum() == 0:
                continue
            Xs, K = gram(Xf[idx], w, fit_dev)
            te = np.where(fo == f)[0]
            for lam in LAMBDAS:
                v = solve(Xs, K, w, t, lam)
                vs[(variant, lam, f)] = v
                held[lam][te] = (Xf[te] @ v).numpy()
            del Xs, K
        for lam in LAMBDAS:
            r = held[lam]
            ok = ~np.isnan(r)
            s = stats(r[ok & err], r[ok & np.isin(cls, HARD)], r[ok & np.isin(cls, NEG)])
            s.update(variant=variant, lam=lam, by_fold={f: round(float(np.mean(r[ok & err & (fo == f)])), 3) if (ok & err & (fo == f)).any() else None for f in (0, 1)},
                     at_correct_closes=round(float(np.mean(r[ok & okc])), 3) if (ok & okc).any() else None,
                     at_ERR2=round(float(np.mean(r[ok & (cls == "ERR2")])), 3) if (ok & (cls == "ERR2")).any() else None)
            grid.append(s)
            ex.log("held-out", s)
    out["grid"] = grid
    passing = [g for g in grid if g["hard_p99"] is not None and g["hard_p99"] < 0.1 and g["pos_mean"] >= 0.5]
    best = max(passing or (grid if SMOKE else []), key=lambda g: g["pos_mean"], default=None)
    out["chosen"] = best
    ex.save_json("fit.json", out)
    if best is None:
        ex.result(fit={k: v for k, v in out.items() if k != "grid"}, adapters=[],
                  headline="no variant and ridge strength passed the selection rule; no neuron built")
        raise SystemExit(0)
    variant, lam = best["variant"], best["lam"]
    idx, t, w = design(cls, variant, use)
    Xs, K = gram(Xf[idx], w, fit_dev)
    v_all = solve(Xs, K, w, t, lam)
    del Xs, K
    v_fold = {f: vs[(variant, lam, f)] for f in (0, 1) if (variant, lam, f) in vs}

    # ---- the write, calibrated on the model ----
    rms_y, rms_h = float(np.mean(rec["rms_y"][err])), float(np.mean(rec["rms_h"][err]))
    repro = float((rec["top"][err] == BTC).mean())
    target = 2.0 if repro >= 0.5 else 6.0  # if the bf16 model does not repeat the fault, the served model's deficit is unknown: more headroom
    c = float(np.clip(np.median(gap) + 3, 3, 9))
    with torch.no_grad():
        u_unit = C.write_vector(model, layer, Q, [BTC, TICK], 1.0, rms_y, rms_h).detach()
    edit = Edit(layers(model)[layer].mlp.down_proj)
    faulty = [r for r in rows if r["fault"]]
    ridx = {id(r): n for n, r in enumerate(rows)}
    key1 = {(n, p): k for k, (n, p) in enumerate(zip(rec["row"], rec["pos"]))}
    cal = []
    for it in range(3):  # the write passes through two RMS norms and a soft cap, so its size is measured, not assumed
        edit.set(u_unit * c, v_all, dev)
        rec2, _ = run_rows(model, faulty, layer, dev, False, ex, f"pass 2 (calibration {it + 1}, c {c:.2f})")
        e2 = np.array(rec2["cls"]) == "ERR"
        pair = [(key1[(ridx[id(faulty[n])], p)], k) for k, (n, p) in enumerate(zip(rec2["row"], rec2["pos"])) if e2[k] and (ridx[id(faulty[n])], p) in key1]
        i1, i2 = np.array([a for a, _ in pair]), np.array([b for _, b in pair])
        shift = (rec2["lq"][i2] - rec2["lb"][i2]) - (rec["lq"][i1] - rec["lb"][i1])
        margin = rec2["lq"][i2] - rec2["lother"][i2]
        cal.append(dict(c=round(c, 2), shift_median=round(float(np.median(shift)), 2), margin_median=round(float(np.median(margin)), 2),
                        margin_p10=round(float(np.quantile(margin, 0.1)), 2), close_is_top=int((margin > 0).sum()), n=int(len(margin))))
        ex.log("calibration", cal[-1])
        need = target - float(np.quantile(margin, 0.1))
        if need <= 0 or c >= 12:
            break
        c = float(min(12.0, c * (1 + need / max(float(np.median(shift)), 0.5))))
    edit.set()
    c1 = c
    u1 = u_unit * c1
    out["write"] = dict(rms_y=round(rms_y, 4), rms_h=round(rms_h, 4), bf16_repeats_the_fault=round(repro, 3), target_margin_p10=target, calibration=cal,
                        c=round(c1, 2), v_norm=round(float(v_all.norm()), 4), u_norm=round(float(u1.norm()), 4))
    ex.log("write:", out["write"])
    ex.save_json("fit.json", out)

    # ---- pass 3: held-out neurons (each row with the neuron fitted without its task) at the calibrated strength ----
    good = [r for r in rows if not r["fault"]]
    test = faulty + good[:GOOD_TEST]
    parts = []
    for f in (0, 1):
        sub = [r for r in test if fold(r["iid"]) == f]
        if not sub or f not in v_fold:
            continue
        edit.set(u1, v_fold[f], dev)
        r3, _ = run_rows(model, sub, layer, dev, False, ex, f"pass 3 (held-out fold {f})")
        ref = {(ridx[id(sub[n])], p): k for k, (n, p) in enumerate(zip(r3["row"], r3["pos"]))}
        for kk, k3 in ref.items():
            if kk in key1:
                parts.append((r3["cls"][k3], int(rec["top"][key1[kk]]), int(r3["top"][k3]), float(r3["lq"][k3] - r3["lother"][k3])))
    edit.set()
    ho = {}
    for c in sorted({p[0] for p in parts}):
        ps = [p for p in parts if p[0] == c]
        ho[c] = dict(n=len(ps), top_changed=sum(p[1] != p[2] for p in ps), top_is_close_before=sum(p[1] == Q for p in ps),
                     top_is_close_after=sum(p[2] == Q for p in ps))
    negs = [p for p in parts if p[0] in NEG]
    errs = [p for p in parts if p[0] == "ERR"]
    out["held_out_edit"] = dict(by_class=ho, ERR_flipped_to_close=sum(p[2] == Q for p in errs), ERR_n=len(errs),
                                ERR_margin_median=round(float(np.median([p[3] for p in errs])), 2) if errs else None,
                                negatives_changed=sum(p[1] != p[2] for p in negs), negatives_n=len(negs))
    ex.log("held-out edit:", out["held_out_edit"])
    ex.save_json("fit.json", out)
    g1 = len(errs) > 0 and sum(p[2] == Q for p in errs) >= 0.7 * len(errs)
    g2 = len(negs) > 0 and sum(p[1] != p[2] for p in negs) <= 0.001 * len(negs)
    out["gates"] = dict(ERR_flipped_at_least_70_percent=bool(g1), negatives_changed_at_most_1_in_1000=bool(g2))

    # ---- generation check from each distinct faulty state ----
    seen, states = set(), []
    for r in faulty:
        k = (r["iid"], r["call_chars"], r["fault"])
        if k not in seen and fold(r["iid"]) in v_fold:
            seen.add(k)
            states.append(r)
    gen = []
    stop = [TC1, TR] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    for r in states[:GEN_STATES]:
        i = next(p for p, c in r["labels"] if c in ("ERR", "ERR2"))
        s = max(k for k in range(i) if r["output_ids"][k] == TC0)
        prefix = torch.tensor([r["prompt_ids"] + r["output_ids"][:i]], device=dev)
        row = dict(iid=r["iid"], fault=r["fault"], served_token=r["output_ids"][i])
        for arm in ("unedited", "edited"):
            edit.set(*((u1, v_fold[fold(r["iid"])], dev) if arm == "edited" else ()))
            try:
                with torch.no_grad():
                    g = model.generate(prefix, max_new_tokens=GEN_NEW, do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id or 0, use_cache=True)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                row[arm] = dict(error="out of memory")
                continue
            new = g[0, prefix.shape[1]:].tolist()
            call = r["output_ids"][s + 1:i] + [x for x in new if x not in (TC1, TR)]
            row[arm] = dict(first_token=new[0] if new else None, new_tokens=len(new), ended=bool(new) and new[-1] in stop,
                            well_formed=well_formed(call, txt))
        edit.set()
        gen.append(row)
        ex.log("generation", row)
    done = [x for x in gen if "error" not in x.get("unedited", {}) and "error" not in x.get("edited", {})]
    out["generation"] = dict(states=len(gen), unedited_repeats_the_served_token=sum(x["unedited"]["first_token"] == x["served_token"] for x in done),
                             unedited_well_formed=sum(x["unedited"]["well_formed"] for x in done), edited_first_token_is_close=sum(x["edited"]["first_token"] == Q for x in done),
                             edited_well_formed=sum(x["edited"]["well_formed"] for x in done), rows=gen)
    ex.save_json("fit.json", out)

    # ---- adapters ----
    saved = []
    if (g1 and g2) or SMOKE:
        torch.manual_seed(140)
        rnd = torch.randn_like(v_all)
        rnd = rnd / rnd.norm() * v_all.norm()
        out["control_readout_at_ERR"] = round(float((Xf[err] @ rnd).mean()), 4)
        base_name = "google/gemma-4-31B-it"
        saved = [save_adapter("delim", layer, u1, v_all, base_name), save_adapter("delim_control", layer, u1, rnd, base_name)]
    ex.save_json("fit.json", out)
    ex.result(fit={k: v for k, v in out.items() if k not in ("grid", "generation")}, generation={k: v for k, v in out["generation"].items() if k != "rows"},
              adapters=[os.path.basename(s) for s in saved],
              headline=(f"31B delimiter neuron ({'saved' if saved else 'gates not passed, not saved'}): held-out read-out at faulty string ends "
                        f"{best['pos_mean']} (hard-negative 99th percentile {best['hard_p99']}); with held-out neurons the closing token is the "
                        f"most probable at {out['held_out_edit']['ERR_flipped_to_close']} of {out['held_out_edit']['ERR_n']} faulty states and "
                        f"{out['held_out_edit']['negatives_changed']} of {out['held_out_edit']['negatives_n']} other states change; greedy "
                        f"continuations well formed {out['generation']['edited_well_formed']} of {len(done)} edited against "
                        f"{out['generation']['unedited_well_formed']} unedited; c {round(c1, 2)}"))

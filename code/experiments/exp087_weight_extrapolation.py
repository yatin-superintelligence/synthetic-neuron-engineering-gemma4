"""EXP-087: weight extrapolation (ExPO) for E2B, training-free: push the instruct weights further from the pretrained base.

ExPO (Zheng et al. 2024, arXiv 2404.16792): theta = theta_instruct + alpha * (theta_instruct - theta_base). Reported up to +6.8% on
AlpacaEval 2.0 for models from 1.8B to 70B, with no training; never tested on code or agents as far as the 2026-11-08 survey found.
Here the anchor is the model we have used throughout as the bf16 reference, the public QAT unquantized E2B (closest to the served
4-bit model), and the direction is google/gemma-4-E2B-it minus google/gemma-4-E2B (the model card of the instruct model names the
latter as its base): theta = theta_qat + alpha * (theta_it - theta_base), for every language-model tensor (vision and audio towers
unchanged). Nothing is trained.

Pre-registered (D-059, before any result):
  Scan, MBPP+ dev half (189), greedy, no thinking, answers capped at 4,096 new tokens (as in EXP-086, D-057), bf16 on the official
    vLLM 0.19.1: alpha = 0 (the QAT unquantized folder itself), 0.1, 0.2, 0.3, 0.5.
  Selection: the alpha > 0 with most dev passes (ties: smaller alpha).
  Confirmation, MBPP+ test half (189), up to 8,192 new tokens: the selected alpha, alpha = 0, and a control edit of matched
    strength (each tensor moved by a random direction with the same Frobenius norm as its alpha * delta, seed 87), at bf16; then
    alpha = 0 and the selected alpha with vLLM's dynamic FP8 quantization (the quantized check; no 4-bit quantizer is available).
  Gate for the next stage (a low-rank LoRA approximation on the official W4A16 checkpoint, then official agent runs): the selected
    alpha beats alpha = 0 on the test half at bf16 (exact McNemar, two-sided, p < 0.05), beats the control edit, and is not lower
    at FP8. Otherwise the scan is the result.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, bench  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.vllm_eval import evaluate, mcnemar  # noqa: E402

ALPHAS = [0.1, 0.2, 0.3, 0.5]
SCAN_MAX = 4096
REPOS = dict(qat="google/gemma-4-E2B-it-qat-q4_0-unquantized", it="google/gemma-4-E2B-it", base="google/gemma-4-E2B")
WORK = "/content/expo"
OUT = os.path.join(ROOT, "artifacts_local", "EXP-087")


def download():
    from huggingface_hub import snapshot_download
    return {k: snapshot_download(r, local_dir=f"/content/hf_{k}") for k, r in REPOS.items()}


def shared_k_norm(qat_dir):
    """vLLM 0.19.1 requires self_attn.k_norm.weight in every layer of an unquantized Gemma 4 checkpoint, but the public bf16
    checkpoint has it only in the 15 layers that compute keys; the 20 KV-shared layers (15 to 34) reuse the keys of layer 13
    (sliding) or 14 (full) and never apply their own k_norm (the official W4A16 checkpoint lacks them too and loads, because the
    check is skipped for quantized models). Placeholders: each shared layer gets the k_norm of the layer whose keys it reuses."""
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(qat_dir, "config.json"), encoding="utf-8"))
    types = cfg.get("text_config", cfg)["layer_types"]
    first = len(types) - cfg.get("text_config", cfg)["num_kv_shared_layers"]
    src = {t: max(i for i in range(first) if types[i] == t) for t in set(types[:first])}
    out = {}
    with safe_open(os.path.join(qat_dir, "model.safetensors"), "pt") as s:
        for i in range(first, len(types)):
            out[f"model.language_model.layers.{i}.self_attn.k_norm.weight"] = s.get_tensor(f"model.language_model.layers.{src[types[i]]}.self_attn.k_norm.weight").clone()
    return out


def build(paths, dst, alpha, control_seed=None):
    """Writes theta_qat + alpha * (theta_it - theta_base); with control_seed, a random direction of the same norm per tensor."""
    import shutil
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in os.listdir(paths["qat"]):
        p = os.path.join(paths["qat"], f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copy2(p, dst)
    files = sorted(f for f in os.listdir(paths["qat"]) if f.endswith(".safetensors"))
    assert files == ["model.safetensors"], files
    g = torch.Generator().manual_seed(control_seed) if control_seed is not None else None
    rep = dict(moved=0, kept=0, missing=[], rel_change=[])
    out = {}
    with safe_open(os.path.join(paths["qat"], files[0]), "pt") as q, safe_open(os.path.join(paths["it"], "model.safetensors"), "pt") as it, \
            safe_open(os.path.join(paths["base"], "model.safetensors"), "pt") as b:
        names_it, names_b = set(it.keys()), set(b.keys())
        for n in q.keys():
            t = q.get_tensor(n)
            if "language_model" not in n or n not in names_it or n not in names_b or not t.is_floating_point():
                out[n] = t
                rep["kept"] += 1
                if "language_model" in n and (n not in names_it or n not in names_b):
                    rep["missing"].append(n)
                continue
            ti, tb = it.get_tensor(n), b.get_tensor(n)
            if ti.shape != t.shape or tb.shape != t.shape:
                out[n] = t
                rep["missing"].append(n + " (shape)")
                continue
            # row chunks keep float32 copies of the 262,144 x 8,960 per-layer embedding under about 1 GB each
            step = max(1, int(2.5e8 // max(1, t[0].numel()))) if t.dim() > 1 else t.shape[0]
            new = torch.empty_like(t)
            dn2 = tn2 = rn2 = 0.0
            chunks = []
            for k in range(0, t.shape[0], step):
                d = (ti[k:k + step].float() - tb[k:k + step].float()) * alpha
                dn2 += float(d.pow(2).sum())
                tn2 += float(t[k:k + step].float().pow(2).sum())
                if g is not None:
                    r = torch.randn(d.shape, generator=g)
                    rn2 += float(r.pow(2).sum())
                    chunks.append(r.to(torch.bfloat16))
                else:
                    new[k:k + step] = (t[k:k + step].float() + d).to(t.dtype)
            if g is not None:  # control: the random direction rescaled to the norm of alpha * delta for this tensor
                scale = (dn2 ** 0.5) / max(rn2 ** 0.5, 1e-12)
                for i, k in enumerate(range(0, t.shape[0], step)):
                    new[k:k + step] = (t[k:k + step].float() + chunks[i].float() * scale).to(t.dtype)
            rep["rel_change"].append((dn2 ** 0.5) / max(tn2 ** 0.5, 1e-12))
            out[n] = new
            rep["moved"] += 1
    added = 0
    for n, t in shared_k_norm(paths["qat"]).items():
        if n not in out:
            out[n] = t.to(torch.bfloat16).clone()  # one storage per tensor: safetensors refuses shared memory
            added += 1
    assert added == 20, f"expected 20 k_norm placeholders, added {added}"
    rep["k_norm_placeholders_added"] = added
    save_file(out, os.path.join(dst, files[0]), metadata={"format": "pt"})
    rc = sorted(rep.pop("rel_change"))
    rep["rel_change_median"] = round(rc[len(rc) // 2], 5) if rc else None
    rep["rel_change_max"] = round(rc[-1], 5) if rc else None
    rep["missing"] = rep["missing"][:20]
    return rep


with Experiment("EXP-087", "weight-extrapolation", category="improvement", depends_on=["EXP-086"],
                motivation="Training-free ways to add capability found by the 2026-11-08 literature survey; ExPO needs only the public base and instruct "
                           "checkpoints, and its result is one weight delta.",
                hypothesis="Some alpha in 0.1 to 0.5 raises E2B's MBPP+ test-half pass count against alpha = 0 (exact McNemar p < 0.05) and against a "
                           "control edit of matched strength, and holds at FP8.",
                config=dict(alphas=ALPHAS, repos=REPOS, scan="MBPP+ dev 189, cap 4,096", confirm="MBPP+ test 189, 8,192, bf16 and FP8",
                            control="random direction per tensor, same Frobenius norm as alpha * delta, seed 87",
                            gate="McNemar p < 0.05 on test at bf16, above control, not lower at FP8")) as ex:
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    t0 = time.time()
    paths = download()
    ex.log("downloaded", paths, f"{time.time() - t0:.0f} s")
    tok = AutoTokenizer.from_pretrained(paths["qat"])
    dev, test = bench.mbppplus("dev"), bench.mbppplus("test")
    alphas = ALPHAS
    if SMOKE:
        dev, test, alphas = dev[:6], test[:6], ALPHAS[:1]
    res = {}
    reports = {}
    A0 = os.path.join(WORK, "a0")  # alpha = 0: the QAT unquantized weights unchanged, plus the official k_norm tensors vLLM requires
    reports["alpha0"] = build(paths, A0, 0.0)
    ex.log("alpha0 built", reports["alpha0"])
    res["alpha0"], _ = evaluate(A0, "alpha0_dev", dev, tok, ex, OUT, SCAN_MAX)
    for a in alphas:
        tag = f"alpha{a}"
        t0 = time.time()
        reports[tag] = build(paths, os.path.join(WORK, "m"), a)
        ex.log(tag, "built", f"{time.time() - t0:.0f} s", reports[tag])
        res[tag], _ = evaluate(os.path.join(WORK, "m"), tag + "_dev", dev, tok, ex, OUT, SCAN_MAX)
        json.dump(dict(res=res, reports=reports), open(os.path.join(OUT, "scan.json"), "w"), indent=1)
    best = max((f"alpha{a}" for a in alphas), key=lambda t: (res[t]["passed"], -float(t[5:])))
    ba = float(best[5:])
    ex.log("selected", best, res[best]["passed"], "dev; alpha0", res["alpha0"]["passed"])
    ids = [t["task_id"] for t in test]
    conf = {}
    conf["alpha0"], p0 = evaluate(A0, "alpha0_test", test, tok, ex, OUT)
    build(paths, os.path.join(WORK, "m"), ba)
    conf[best], pb = evaluate(os.path.join(WORK, "m"), best + "_test", test, tok, ex, OUT)
    reports["control"] = build(paths, os.path.join(WORK, "c"), ba, control_seed=87)
    conf["control"], pc = evaluate(os.path.join(WORK, "c"), "control_test", test, tok, ex, OUT)
    x, y, p = mcnemar([pb[i] for i in ids], [p0[i] for i in ids])
    cx, cy, cp = mcnemar([pb[i] for i in ids], [pc[i] for i in ids])
    stats = dict(best_only=x, alpha0_only=y, p=round(p, 4), best_vs_control=dict(best_only=cx, control_only=cy, p=round(cp, 4)))
    ex.log("test half bf16", {t: r["passed"] for t, r in conf.items()}, stats)
    # quantized check: vLLM dynamic FP8 on the same two models
    fp8 = {}
    try:
        fp8["alpha0_fp8"], f0 = evaluate(A0, "alpha0_fp8_test", test, tok, ex, OUT, extra_args=("--quantization", "fp8"))
        fp8[best + "_fp8"], fb = evaluate(os.path.join(WORK, "m"), best + "_fp8_test", test, tok, ex, OUT, extra_args=("--quantization", "fp8"))
        fx, fy, fpv = mcnemar([fb[i] for i in ids], [f0[i] for i in ids])
        stats["fp8"] = dict(best_only=fx, alpha0_only=fy, p=round(fpv, 4))
    except Exception as e:  # noqa: BLE001
        ex.log("FP8 check FAILED", repr(e)[:1500])
        stats["fp8"] = dict(error=repr(e)[:500])
    gate = stats["p"] < 0.05 and x > y and cx > cy and stats["fp8"].get("best_only", 0) >= stats["fp8"].get("alpha0_only", 1)
    ex.result(scan={t: (r["passed"], r["categories"]) for t, r in res.items()}, builds=reports, selected=best,
              test_bf16={t: r["passed"] for t, r in conf.items()}, test_fp8={t: r["passed"] for t, r in fp8.items()}, stats=stats, gate_passed=gate,
              headline=f"selected {best}: dev {res[best]['passed']} vs {res['alpha0']['passed']} of {len(dev)}; test bf16 {conf[best]['passed']} vs "
                       f"{conf['alpha0']['passed']} (control {conf['control']['passed']}) of {len(test)}, McNemar p = {stats['p']}; gate {'passed' if gate else 'not passed'}")

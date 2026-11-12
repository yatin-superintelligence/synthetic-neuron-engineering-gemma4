"""EXP-086b: EXP-086's pre-registered bf16 precision check, rerun after a loading fault.

EXP-086 (D-056) ends with the selected block and the unedited model on MBPP+ test at bf16 (the public QAT unquantized checkpoint).
That step cannot load in vLLM 0.19.1: it requires self_attn.k_norm.weight in all layers, and the public bf16 checkpoint omits it
in the 20 KV-shared layers (found by EXP-087, 2026-11-08). EXP-086 catches the error and records the check as failed. This script
runs the same check with the fix: placeholders for the missing k_norm tensors, which KV-shared layers never apply (the
k_norm of layer 13 or 14, whose keys they reuse). The selected block is read from EXP-086's results; nothing else changes.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, bench  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.vllm_eval import evaluate, mcnemar  # noqa: E402

src = open(os.path.join(os.path.dirname(__file__), "exp086_layer_duplication.py"), encoding="utf-8").read()
g = {"__file__": os.path.join(os.path.dirname(__file__), "exp086_layer_duplication.py"), "__name__": "exp086"}
exec(compile(src.split('if __name__ == "__main__" and "--dry" in sys.argv:')[0], "exp086", "exec"), g)  # order_for, build
OUT = os.path.join(ROOT, "artifacts_local", "EXP-086b")


def add_k_norm(folder, order, hf):
    """Placeholders for k_norm in KV-shared layers (unused there; vLLM requires them in an unquantized checkpoint): the k_norm of
    the last non-shared layer of the same attention type (13 sliding, 14 full), which supplies those layers' keys (EXP-087)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    cfg = json.load(open(os.path.join(hf, "config.json"), encoding="utf-8"))
    types = cfg.get("text_config", cfg)["layer_types"]
    first = len(types) - cfg.get("text_config", cfg)["num_kv_shared_layers"]
    with safe_open(os.path.join(hf, "model.safetensors"), "pt") as s:
        kn = {t: s.get_tensor(f"model.language_model.layers.{max(i for i in range(first) if types[i] == t)}.self_attn.k_norm.weight")
              for t in set(types[:first])}
    path = os.path.join(folder, "model.safetensors")
    sd = load_file(path)
    added = 0
    for p, j in enumerate(order):
        n = f"model.language_model.layers.{p}.self_attn.k_norm.weight"
        if n not in sd:
            sd[n] = kn[types[j]].to(torch.bfloat16).clone()  # one storage per tensor: safetensors refuses shared memory
            added += 1
    save_file(sd, path, metadata={"format": "pt"})
    return added


with Experiment("EXP-086b", "layer-duplication-bf16-check", category="improvement", depends_on=["EXP-086"],
                motivation="EXP-086's pre-registered bf16 check could not load (missing k_norm in KV-shared layers of the public bf16 checkpoint).",
                hypothesis="As pre-registered in D-056: the selected block is not lower than the unedited model on MBPP+ test at bf16.",
                config=dict(fix="k_norm of KV-shared layers copied from the official W4A16 checkpoint")) as ex:
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(glob.glob(os.path.join(ROOT, "experiments", "EXP-086_*", "results.json"))[0], encoding="utf-8"))
    res = res.get("results", res)
    best = res["selected"]
    bs, bk = int(best.split("_s")[1].split("_")[0]), int(best.split("_k")[1])
    ex.log("selected block from EXP-086:", best)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    tok = AutoTokenizer.from_pretrained(hf)
    test = bench.mbppplus("test")
    if SMOKE:
        test = test[:6]
    ids = [t["task_id"] for t in test]
    out = {}
    for tag, order in (("official_bf16", g["order_for"](None, 0)), (best + "_bf16", g["order_for"](bs, bk))):
        d = os.path.join("/content/dup86b", tag)
        g["build"](hf, d, order, bs >= 15 and tag != "official_bf16")
        n = add_k_norm(d, order, hf)
        ex.log(tag, "built, k_norm tensors added:", n)
        out[tag] = evaluate(d, tag + "_test", test, tok, ex, OUT)
    (so, po), (sb, pb) = out["official_bf16"], out[best + "_bf16"]
    x, y, p = mcnemar([pb[i] for i in ids], [po[i] for i in ids])
    ex.result(selected=best, test_bf16={"official": so["passed"], best: sb["passed"]}, stats=dict(best_only=x, official_only=y, p=round(p, 4)),
              headline=f"bf16 MBPP+ test: {best} {sb['passed']} vs unedited {so['passed']} of {len(test)} (McNemar p = {round(p, 4)})")

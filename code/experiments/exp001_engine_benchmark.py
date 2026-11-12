"""EXP-001: choose the fastest inference engine for Gemma 4 E2B on the Intel Arc iGPU, at both precisions.

Engines: llama.cpp b12129 Vulkan, llama.cpp b12129 SYCL, PyTorch 2.15 XPU eager (HF transformers 5.22).
Precisions: Google QAT q4_0 GGUF, and the QAT full-precision checkpoint converted to F16 GGUF (BF16 for torch).
Metrics: prompt tok/s, single-stream decode tok/s, aggregate decode tok/s at 1/4/8/16/32 parallel sequences.
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT  # noqa: E402
from sne.registry import Experiment  # noqa: E402

GG = os.path.join(ROOT, "models", "gguf")
MODELS = {"q4_0": os.path.join(GG, "gemma-4-E2B_q4_0-it.gguf"), "f16": os.path.join(GG, "e2b-qat-f16.gguf")}


def run(cmd, ex, timeout=1800):
    ex.log("CMD", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    rows = [l for l in out.splitlines() if l.startswith("|")]
    for l in rows:
        ex.log(l)
    return out, rows


def parse_bench(rows):
    res = {}
    for l in rows:
        c = [x.strip() for x in l.strip("|").split("|")]
        if len(c) >= 7 and c[-2] in ("pp512", "tg128"):
            res[c[-2]] = float(re.match(r"([0-9.]+)", c[-1]).group(1))
    return res


def parse_batched(rows):
    res = {}
    for l in rows:
        c = [x.strip() for x in l.strip("|").split("|")]
        if len(c) == 10 and c[0].isdigit():
            res[int(c[2])] = dict(pp_tps=float(c[5]), tg_tps=float(c[7]), total_tps=float(c[9]))
    return res


with Experiment("EXP-001", "inference-engine-benchmark", category="infrastructure",
                motivation="Hundreds of generation-based evaluations are planned; throughput on the Arc iGPU decides "
                           "how many experiments and seeds are affordable. The engine must also support per-request "
                           "LoRA adapters so edits can be evaluated without reloading the base model.",
                hypothesis="llama.cpp (Vulkan or SYCL) is several times faster than PyTorch XPU eager generation; "
                           "q4_0 decodes faster than F16 because decode is memory-bandwidth bound.",
                config=dict(models=MODELS, llama_cpp="b12129", ngl=99, pp=256, tg=256, npl=[1, 4, 8, 16, 32],
                            flash_attn="on")) as ex:
    out = {}
    for backend in ["vulkan", "sycl"]:
        for prec, path in MODELS.items():
            key = f"llamacpp_{backend}_{prec}"
            exe = os.path.join(ROOT, "tools", "llama", backend)
            _, rows = run([os.path.join(exe, "llama-bench.exe"), "-m", path, "-ngl", "99", "-p", "512", "-n", "128",
                           "-r", "2", "-fa", "1"], ex)
            single = parse_bench(rows)
            _, rows = run([os.path.join(exe, "llama-batched-bench.exe"), "-m", path, "-ngl", "99", "-c", "17000",
                           "-b", "2048", "-ub", "512", "-npp", "256", "-ntg", "256", "-npl", "1,4,8,16,32",
                           "-fa", "on"], ex)
            out[key] = dict(single=single, batched=parse_batched(rows))
            ex.result(**out)
    # PyTorch XPU eager reference (bf16 QAT checkpoint), short runs only (it is far slower).
    code = r"""
import sys, time, torch; sys.path.insert(0, r'%s')
from sne.model import load
from sne.evaluate import render
m, tok = load(r'%s')
p = render(tok, ['Write a Python function that returns the n-th Fibonacci number.'] * 8)
res = {}
for bs in (1, 8):
    enc = tok(p[:bs], return_tensors='pt', padding=True, add_special_tokens=True).to('xpu')
    with torch.no_grad():
        m.generate(**enc, max_new_tokens=4, do_sample=False)
        torch.xpu.synchronize(); t = time.time()
        o = m.generate(**enc, max_new_tokens=32, min_new_tokens=32, do_sample=False)
        torch.xpu.synchronize(); dt = time.time() - t
    res[bs] = round(32 * bs / dt, 2)
print('TORCH_RESULT', res)
""" % (os.path.join(ROOT, "src"), os.path.join(ROOT, "models", "gemma-4-E2B-it-qat-unquantized"))
    t = time.time()
    p = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True, timeout=3600,
                       encoding="utf-8", errors="replace")
    m = re.search(r"TORCH_RESULT (\{.*\})", p.stdout)
    out["torch_xpu_bf16_eager"] = dict(decode_tps_by_batch=eval(m.group(1)) if m else None,
                                       error=None if m else p.stderr[-800:], wall_s=round(time.time() - t, 1))
    ex.log("torch", out["torch_xpu_bf16_eager"])
    best = max(((k, v["batched"].get(16, {}).get("tg_tps", 0)) for k, v in out.items() if k.startswith("llamacpp")),
               key=lambda kv: kv[1])
    ex.result(**out, headline=f"best aggregate decode @16 seqs: {best[0]} {best[1]} tok/s; "
                              f"torch eager bs1/bs8: {out['torch_xpu_bf16_eager']['decode_tps_by_batch']}")

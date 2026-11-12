"""EXP-007: OpenVINO engines vs the current Vulkan choice, on the same machine and power mode.

Arms: llama.cpp b12129 OpenVINO backend (GPU) on the same Q4_0 and F16 GGUFs, using llama-batched-bench; OpenVINO GenAI
2026.4.1 with OpenVINO/gemma-4-E2B-it-int4-ov (single stream and 8 concurrent if the pipeline allows it). Vulkan is
re-measured in the same session as the reference, so power mode and thermal state match.
Decision rule: switch only if aggregate decode at 8 parallel beats Vulkan by >20% and per-request LoRA still works.
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import MODELS, ROOT  # noqa: E402
from sne.registry import Experiment  # noqa: E402

GG = os.path.join(MODELS, "gguf")
FILES = {"q4_0": os.path.join(GG, "gemma-4-E2B_q4_0-it.gguf"), "f16": os.path.join(GG, "e2b-qat-f16.gguf")}
OV_DIR = os.path.join(MODELS, "ov-gemma-4-E2B-it-int4")
OV_VENV = os.path.join(ROOT, ".venv-ov", "Scripts", "python.exe")
GENAI = r"""
import json, sys, time
import openvino_genai as g
path = sys.argv[1]
pipe = g.VLMPipeline(path, "GPU")
cfg = g.GenerationConfig(); cfg.max_new_tokens = 128; cfg.min_new_tokens = 128; cfg.do_sample = False
prompt = "Write a Python function that returns the n-th Fibonacci number."
res = {}
pipe.generate(prompt, generation_config=cfg)
t = time.time(); out = pipe.generate(prompt, generation_config=cfg); dt = time.time() - t
res["single_tps_wall"] = round(128 / dt, 2)
try:
    pm = out.perf_metrics
    res["tps_metric"] = round(pm.get_throughput().mean, 2)
    res["ttft_ms"] = round(pm.get_ttft().mean, 1)
except Exception as e:
    res["metric_error"] = repr(e)
print("RESULT " + json.dumps(res))
"""


def batched(backend, path, ex, env=None):
    exe = os.path.join(ROOT, "tools", "llama", backend, "llama-batched-bench.exe")
    cmd = [exe, "-m", path, "-ngl", "99", "-c", "9000", "-b", "2048", "-ub", "512", "-npp", "256", "-ntg", "128",
           "-npl", "1,8", "-fa", "on"]
    ex.log("CMD", " ".join(cmd))
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, encoding="utf-8", errors="replace",
                       env=dict(os.environ, **(env or {})))
    res = {}
    for l in (p.stdout + p.stderr).splitlines():
        c = [x.strip() for x in l.strip("|").split("|")]
        if l.startswith("|") and len(c) == 10 and c[0].isdigit():
            res[c[2]] = dict(pp=float(c[5]), tg=float(c[7]))
            ex.log(l)
    if not res:
        ex.log("NO RESULT", (p.stdout + p.stderr)[-1500:])
    return dict(res=res, wall=round(time.time() - t, 1), rc=p.returncode)


with Experiment("EXP-007", "openvino-engine-comparison", category="infrastructure", depends_on=["EXP-001"],
                motivation="The author asked for the fastest engine on the Arc GPU; EXP-001 did not test OpenVINO.",
                hypothesis="OpenVINO GenAI is faster single-stream at int4; the llama.cpp OpenVINO backend is at best on "
                           "par with Vulkan at 8 parallel.",
                config=dict(ov_model="OpenVINO/gemma-4-E2B-it-int4-ov", llama="b12129", npp=256, ntg=128, npl=[1, 8])) as ex:
    out = {}
    for prec, f in FILES.items():
        out[f"vulkan_{prec}"] = batched("vulkan", f, ex)
        out[f"llama_openvino_{prec}"] = batched("openvino", f, ex, env=dict(OPENVINO_DEVICE="GPU"))
        ex.result(**out)
    if not os.path.isdir(OV_DIR):
        dl = ("import truststore; truststore.inject_into_ssl()\nfrom huggingface_hub import snapshot_download\n"
              f"print(snapshot_download('OpenVINO/gemma-4-E2B-it-int4-ov', revision='81a7dc381261ebcea9aac4849aac35496475b389', local_dir=r'{OV_DIR}'))")
        subprocess.run([OV_VENV, "-c", dl], check=False, capture_output=True, text=True)
    p = subprocess.run([OV_VENV, "-c", GENAI, OV_DIR], capture_output=True, text=True, timeout=1800, encoding="utf-8",
                       errors="replace")
    m = re.search(r"RESULT (\{.*\})", p.stdout)
    out["openvino_genai_int4"] = json.loads(m.group(1)) if m else dict(error=(p.stdout + p.stderr)[-1500:])
    ex.log(out["openvino_genai_int4"])
    ex.result(**out, headline="; ".join(f"{k}: {v.get('res', {}).get('8', {}).get('tg', v)}" for k, v in out.items()))

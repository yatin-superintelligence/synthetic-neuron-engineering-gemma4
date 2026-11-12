"""EXP-110b launcher (D-152, D-152b): EXP-110's script with HALF=test (train on the test half's 46 officially resolved runs, at most 8
per task). colab_job cannot pass environment variables, so this wrapper sets them and runs the original script unchanged.
D-152b: the training length limit is measured, not assumed. exp110b_memprobe.py runs one forward and backward pass at rising lengths,
each in its own process, and the limit becomes the longest length that fits (at most 32,768). Conversations above it are left out up
front (counted in the log), so no example fails mid-run. The probe results go to artifacts_local/EXP-110b/memprobe.json."""
import json
import os
import runpy
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["HALF"] = "test"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
probe = []
fit = 0
for n in (20480, 24576, 28672, 32768):  # D-152c: 20,480 fitted even before the logits fix (22.3 GB)
    out = subprocess.run([sys.executable, os.path.join(HERE, "exp110b_memprobe.py"), str(n)], capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("PROBE")), f"PROBE {n} ERROR " + (out.stderr or "")[-300:].replace("\n", " "))
    print(line, flush=True)
    probe.append(line)
    if " OK " not in line:
        break
    fit = n
os.makedirs(os.path.join(HERE, "..", "..", "artifacts_local", "EXP-110b"), exist_ok=True)
json.dump(dict(probe=probe, max_len=fit), open(os.path.join(HERE, "..", "..", "artifacts_local", "EXP-110b", "memprobe.json"), "w"), indent=1)
if not fit:
    sys.exit("memory probe: not even 12,288 tokens fit; see memprobe.json")
os.environ["MAXLEN"] = str(fit)
print(f"training length limit {fit} tokens (measured)", flush=True)
runpy.run_path(os.path.join(HERE, "exp110_success_distillation.py"), run_name="__main__")

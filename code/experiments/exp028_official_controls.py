"""EXP-028: official swegemma controls on this machine (Docker sandbox, verify_task from the real harness).

For the 58 small-fix tasks: the reference patch must resolve and an empty patch must not. Failing tasks are written to
data/kaggle/unsound.json. Cross-checked with the public audit by D. Gluzdov (task_soundness.csv).
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, sweagent as A  # noqa: E402
from sne.registry import Experiment  # noqa: E402

K = os.path.join(ROOT, "data", "kaggle")
PY = os.path.join(ROOT, ".venv-harness", "Scripts", "python.exe")
WORKERS = 4


def small_ids():
    def st(p):
        fs = re.findall(r"^\+\+\+ b/(\S+)", p, re.M)
        return len(fs), sum(1 for l in p.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    t = [json.loads(l) for l in open(os.path.join(K, "tasks.jsonl"), encoding="utf-8")]
    return [x["instance_id"] for x in t if st(x["patch"])[0] == 1 and st(x["patch"])[1] <= 10]


def build_image(log):
    if subprocess.run(["docker", "image", "inspect", "swebench-sandbox:latest"], capture_output=True).returncode == 0:
        return
    d = tempfile.mkdtemp(prefix="sne_img_")
    for f in ("imp.py", "telnetlib.py", "Dockerfile.public"):
        shutil.copy(os.path.join(K, "docker", "docker", f), d)
    shutil.copytree(os.path.join(K, "wheels"), os.path.join(d, "wheels"))
    p = subprocess.run(["docker", "build", "-f", os.path.join(d, "Dockerfile.public"), "-t", "swebench-sandbox:latest", d],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log(p.stdout[-800:], p.stderr[-800:])
    shutil.rmtree(d, ignore_errors=True)


with Experiment("EXP-028", "official-harness-controls-local", category="benchmark", depends_on=["EXP-002"],
                motivation="Every local real-task score depends on the official harness behaving here as on Kaggle. "
                           "Reference fixes must pass and empty patches must fail, per task.",
                hypothesis="Most fastapi and rich small-fix tasks pass both controls. requests tasks cannot pass in this "
                           "path: its src/ layout is not importable in subprocess mode, so they fail the gold control.",
                config=dict(harness="swegemma 0.2.11", adk="1.36.1", workers=WORKERS,
                            sandbox="subprocess inside the sne-harness Linux container, as on Kaggle")) as ex:
    A.ensure_docker(ex.log)
    ids = small_ids()[:3] if SMOKE else small_ids()
    idp, outp = os.path.join(ex.dir, "ids.json"), os.path.join(ex.dir, "controls.json")
    json.dump(ids, open(idp, "w"))
    rel = os.path.relpath(ex.dir, ROOT).replace(os.sep, "/")
    p = subprocess.run(["docker", "run", "--rm", "-v", f"{ROOT}:/proj", "-w", "/proj", "-e", "PYTHONIOENCODING=utf-8",
                        "sne-harness:latest", "python", "-u", "tools/official_eval.py", f"{rel}/controls.json",
                        f"{rel}/ids.json", str(WORKERS)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    ex.log(p.stdout[-5000:])
    ex.log(p.stderr[-1500:])
    rows = json.load(open(outp))
    drop = sorted(r["id"] for r in rows if not r["sound"])
    if not SMOKE:
        json.dump(drop, open(os.path.join(K, "unsound.json"), "w"))
    pubp = os.path.join(ROOT, "scratch", "public_ds", "task_soundness.csv")
    pub = {r["task_id"]: r for r in csv.DictReader(open(pubp, encoding="utf-8"))}
    agree = sum(r["id"] in pub and (pub[r["id"]]["sound"] == "1") == r["sound"] for r in rows)
    by = {}
    for r in rows:
        k = r["id"].split("_")[0]
        by.setdefault(k, [0, 0])
        by[k][0] += r["sound"]
        by[k][1] += 1
    ex.result(n=len(rows), sound=len(rows) - len(drop), dropped=drop, by_repo=by, agrees_with_public_audit=agree,
              headline=f"{len(rows) - len(drop)}/{len(rows)} sound {by}; agrees with public audit on {agree}/{len(rows)}")

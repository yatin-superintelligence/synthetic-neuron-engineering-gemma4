"""Sandboxed execution of model-generated programs.

Each program runs in a fresh temporary directory, in a separate isolated-mode interpreter (`python -I`, no user
site, no PYTHONPATH), with a minimal environment and a wall-clock timeout. Docker Desktop could not start its
engine on this machine (logged 2026-11-05), so this subprocess sandbox is the default; set SNE_EXEC=docker to
use the container runner in sandbox/ instead.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import ROOT

SANDBOX_DIR = os.path.join(ROOT, "sandbox")
PY = sys.executable  # the project venv interpreter (pinned numpy etc.)
MIN_ENV = {k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP", "PATH", "COMSPEC") if k in os.environ}


def _run_local(job):
    d = tempfile.mkdtemp(prefix="sne_prog_")
    try:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(job["program"])
        try:
            p = subprocess.run([PY, "-I", path], cwd=d, capture_output=True, text=True, timeout=job["timeout"],
                               env=MIN_ENV, encoding="utf-8", errors="replace")
            status = "pass" if p.returncode == 0 else "fail"
            err = (p.stderr or "")[-1500:]
            out = (p.stdout or "")[-2000:]
        except subprocess.TimeoutExpired:
            status, err, out = "timeout", "TimeoutExpired", ""
        return job["id"], {"passed": status == "pass", "status": status, "stderr_tail": err, "stdout_tail": out}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_programs(jobs, timeout=30.0, workers=8):
    """jobs: list of (id, program). Returns {id: {"passed", "status", "stderr_tail"}}."""
    payload = [{"id": str(i), "program": p, "timeout": timeout} for i, p in jobs]
    if not payload:
        return {}
    if os.environ.get("SNE_EXEC", "local") == "docker":
        return _run_docker(payload, workers)
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(_run_local, payload))


def _run_docker(payload, workers):
    image = "sne-sandbox:py311"
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        subprocess.run(["docker", "build", "-t", image, SANDBOX_DIR], check=True)
    work = tempfile.mkdtemp(prefix="sne_exec_")
    try:
        with open(os.path.join(work, "jobs.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        shutil.copy(os.path.join(SANDBOX_DIR, "runner.py"), os.path.join(work, "runner.py"))
        subprocess.run(["docker", "run", "--rm", "--name", f"sne_{uuid.uuid4().hex[:8]}", "--network", "none",
                        "--memory", "4g", "--cpus", str(min(workers, 8)), "--pids-limit", "512",
                        "-e", f"SANDBOX_WORKERS={workers}", "-v", f"{work}:/work", image,
                        "python", "/work/runner.py"], check=True, capture_output=True)
        with open(os.path.join(work, "results.json"), encoding="utf-8") as f:
            return json.load(f)
    finally:
        shutil.rmtree(work, ignore_errors=True)

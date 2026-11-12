"""MBPP+ evaluation on the official vLLM (one model at a time), shared by EXP-086 and later experiments.

Moved here unchanged from experiments/scripts/exp086_layer_duplication.py (2026-11-08), with the output folder as an argument:
Vllm (a vLLM OpenAI server with the LlamaServer completion interface), mcnemar (exact two-sided test on paired pass lists),
capped (scores saved rows as if answers were cut at a token cap) and evaluate (one model on a task list, rows saved to out_dir).
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .evaluate import run_suite

PORT = 8011


class Vllm:
    """The official vLLM, one model at a time, with the same completion interface as sne.engine.LlamaServer."""

    def __init__(self, model_dir, log_path, extra_args=()):
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir, "--served-model-name", "m",
               "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "16384", "--gpu-memory-utilization", "0.88",
               "--enforce-eager", "--max-num-seqs", "192", "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0}), *extra_args]
        self.log_path = log_path
        self.p = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 1200:
            if self.p.poll() is not None:
                raise RuntimeError("vLLM exited while loading: " + open(log_path, encoding="utf-8", errors="replace").read()[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                self.load_s = round(time.time() - t0, 1)
                return
            except Exception:  # noqa: BLE001
                time.sleep(5)
        self.close()
        raise RuntimeError("vLLM did not become ready in 20 min")

    def _one(self, prompt, max_tokens):
        body = json.dumps(dict(model="m", prompt=prompt, max_tokens=max_tokens, temperature=0.0)).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        c = r["choices"][0]
        return dict(text=c["text"], n_gen=r["usage"]["completion_tokens"], n_prompt=r["usage"]["prompt_tokens"],
                    stop_type="limit" if c.get("finish_reason") == "length" else "stop")

    def complete_many(self, prompts, max_tokens, lora_scales=None, sampling=None):
        t0 = time.time()
        with ThreadPoolExecutor(192) as ex:
            gens = list(ex.map(lambda p: self._one(p, max_tokens), prompts))
        wall = time.time() - t0
        n = sum(g["n_gen"] for g in gens)
        return gens, dict(wall_s=round(wall, 1), gen_tokens=n, tok_per_s=round(n / max(wall, 1e-6), 1))

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        time.sleep(5)


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired pass lists a, b: returns (a only, b only, p)."""
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    n = x + y
    if n == 0:
        return x, y, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(x, y) + 1)) / 2 ** n
    return x, y, min(1.0, 2 * tail)


def capped(rows, cap):
    """Scores saved rows as if answers had been cut at `cap` new tokens: a pass needs an answer of at most cap tokens."""
    out = []
    for r in rows:
        r = dict(r)
        if (r.get("n_gen") or 0) > cap:
            r["passed"], r["category"] = False, "over_cap"
        out.append(r)
    n, k = len(out), sum(r["passed"] for r in out)
    cats = {}
    for r in out:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    return dict(n=n, passed=k, pass_rate=round(k / n, 4), categories=cats, scored_from_saved_rows=True, cap=cap,
                mean_gen_tokens=round(sum(min(r["n_gen"] or 0, cap) for r in out) / n, 1), load_s=None, total_s=None), out


def evaluate(model_dir, tag, tasks, tok, ex, out_dir, max_tokens=None, extra_args=()):
    saved = os.path.join(out_dir, f"rows_{tag}.json")
    if max_tokens and os.path.exists(saved):  # a dev configuration already run (at 8,192) before the restart: scored under the cap
        summ, rows = capped(json.load(open(saved, encoding="utf-8")), max_tokens)
        ex.log(f"[dup] {tag}: {summ['passed']}/{summ['n']} passed (from saved rows, cap {max_tokens}) | categories {summ['categories']}")
        return summ, {r["task_id"]: r["passed"] for r in rows}
    srv = Vllm(model_dir, os.path.join(out_dir, f"vllm_{tag}.log"), extra_args)
    try:
        summ, rows = run_suite(srv, tok, tasks, max_tokens=max_tokens)
    finally:
        srv.close()
    summ["load_s"] = srv.load_s
    json.dump(rows, open(saved, "w", encoding="utf-8"))
    ex.log(f"[dup] {tag}: {summ['passed']}/{summ['n']} passed | categories {summ['categories']} | mean new tokens {summ['mean_gen_tokens']}"
           f" | load {summ['load_s']} s, total {summ['total_s']} s")
    return summ, {r["task_id"]: r["passed"] for r in rows}

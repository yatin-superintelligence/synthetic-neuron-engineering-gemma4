"""Generation engine: llama.cpp `llama-server` (Vulkan backend) on the Intel Arc GPU.

Chosen by EXP-001 (engine benchmark). One server process holds a base GGUF plus any number of LoRA adapters;
each request selects adapters/scales, so many edits are evaluated without reloading the base model.
Prompts are rendered with the Hugging Face chat template (same string as the PyTorch path), sent raw to /completion.
Decoding is greedy (temperature 0, top_k 1) unless a sampling config is passed explicitly.
"""
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import ROOT

LLAMA_DIR = os.path.join(ROOT, "tools", "llama")


class LlamaServer:
    def __init__(self, gguf, loras=(), backend="vulkan", port=8091, parallel=8, ctx_per_slot=16384,
                 extra_args=(), log_path=None):
        self.gguf, self.loras, self.port, self.parallel = gguf, list(loras), port, parallel
        exe = os.path.join(LLAMA_DIR, backend, "llama-server.exe")
        args = [exe, "-m", gguf, "-ngl", "999", "--port", str(port), "-np", str(parallel),
                "-c", str(ctx_per_slot * parallel), "-fa", "on", "--no-webui", "-b", "2048", "-ub", "512",
                "--seed", "0", "--metrics", "--cache-ram", "0"]  # no host prompt cache: default 8 GB wastes RAM
        if self.loras:
            args += ["--lora", ",".join(self.loras), "--lora-init-without-apply"]
        args += list(extra_args)
        self.cmd = args
        self.log_path = log_path or os.path.join(ROOT, "scratch", f"llama_server_{port}.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.proc = None

    def __enter__(self):
        self._log = open(self.log_path, "a", encoding="utf-8")
        self._log.write("\n# CMD: " + " ".join(self.cmd) + "\n")
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT)
        t = time.time()
        while time.time() - t < 600:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:  # noqa: BLE001
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited, see {self.log_path}")
            time.sleep(1)
        else:
            raise TimeoutError("llama-server did not become healthy")
        self.load_seconds = time.time() - t
        return self

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()

    # ------------------------------------------------------------------ requests
    def _post(self, path, payload, timeout=1800):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def complete(self, prompt, max_tokens=768, lora_scales=None, sampling=None, n_probs=0):
        payload = dict(prompt=prompt, n_predict=max_tokens, temperature=0.0, top_k=1, top_p=1.0, min_p=0.0,
                       repeat_penalty=1.0, seed=0, cache_prompt=False, n_probs=n_probs)
        if sampling:
            payload.update(sampling)
        if self.loras:
            scales = lora_scales or {}
            payload["lora"] = [{"id": i, "scale": float(scales.get(i, 0.0))} for i in range(len(self.loras))]
        r = self._post("/completion", payload)
        return dict(text=r.get("content", ""), n_gen=r.get("tokens_predicted"), n_prompt=r.get("tokens_evaluated"),
                    stop_type=r.get("stop_type"), truncated=r.get("truncated"),
                    timings=r.get("timings", {}), probs=r.get("completion_probabilities"))

    def complete_many(self, prompts, max_tokens=768, lora_scales=None, sampling=None, workers=None):
        workers = workers or self.parallel
        t = time.time()
        with ThreadPoolExecutor(workers) as ex:
            outs = list(ex.map(lambda p: self.complete(p, max_tokens, lora_scales, sampling), prompts))
        dt = time.time() - t
        ntok = sum(o["n_gen"] or 0 for o in outs)
        return outs, dict(wall_s=round(dt, 2), gen_tokens=ntok, tok_per_s=round(ntok / max(dt, 1e-9), 1))


def strip_bos(prompt):
    return prompt[len("<bos>"):] if prompt.startswith("<bos>") else prompt

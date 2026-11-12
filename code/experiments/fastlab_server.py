"""Fast edit lab: one long-lived official vLLM (W4A16 E2B) that takes new LoRA edits at runtime and scores them on the patch bench.

Why: each bench experiment so far restarted vLLM and ran its arms one after another on 257 states with long answers (about 2.5 h).
Here the server stays up; a job (a JSON file in /content/fastlab/jobs) names up to 7 adapter folders; they are loaded into the running
server (VLLM_ALLOW_RUNTIME_LORA_UPDATING), all arms and the unedited model answer the same states at the same time, and per-state
outcomes go to /content/fastlab/results/<job>.json. Splits (EXP-092's states and its seed-92 split): 'quick' is 96 states drawn from
dev and test (seed 99); 'held' is all 257 dev and test states. Answer caps: 'quick' closes the thought at 1,024 tokens and allows 384
more; 'held' uses EXP-092's 2,048 and 1,024. Outcomes per answer: applies, correct (EXP-092's scoring), a backslash-quote the code
does not have (escape), and a failing old_string copied from the thought (self-copy). Stops when /content/fastlab/STOP exists or
after 6 idle hours.
"""
import glob
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT  # noqa: E402

HERE = os.path.dirname(__file__)
_src = open(os.path.join(HERE, "exp092_patch_neurons.py"), encoding="utf-8").read()
P = {"__file__": os.path.join(HERE, "exp092_patch_neurons.py"), "__name__": "exp092"}
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)
LAB = "/content/fastlab"
PORT = 8017
ESC = '\\"'
CAPS = {"quick": (1024, 384), "held": (2048, 1024)}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def escape_error(text, st):
    m = P["CALL"].search(text or "")
    if not m or m.group(1) != "edit_file" or ESC in st["excerpt"]:
        return False
    a = dict(P["ARG"].findall(m.group(2)))
    return ESC in (a.get("old_string") or "") or ESC in (a.get("new_string") or "")


def self_copy(text, st):
    m = P["CALL"].search(text or "")
    if not m or m.group(1) != "edit_file":
        return False
    a = dict(P["ARG"].findall(m.group(2)))
    old = a.get("old_string") or ""
    if not old or st["excerpt"].count(old) == 1:
        return False
    bad = [l.strip() for l in old.split("\n") if l.strip() and l.strip() not in st["excerpt"]]
    th = text[: text.find("<|tool_call>")]
    return bool(bad) and all(l in th for l in bad)


def req(path, body, timeout=3600):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001  (load/unload answer with plain text)
        return raw.decode(errors="replace")


def answer(model, prompt, caps):
    t1, t2 = caps
    r = req("/v1/completions", dict(model=model, prompt=prompt, max_tokens=t1, temperature=0.0, skip_special_tokens=False))
    c = r["choices"][0]
    t = c["text"]
    if c.get("finish_reason") == "length" and "<channel|>" not in t and "<|tool_call>" not in t:
        r2 = req("/v1/completions", dict(model=model, prompt=prompt + t + "<channel|>", max_tokens=t2, temperature=0.0, skip_special_tokens=False))
        t = t + "<channel|>" + r2["choices"][0]["text"]
    return t


def main():
    os.makedirs(os.path.join(LAB, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(LAB, "results"), exist_ok=True)
    os.makedirs(os.path.join(LAB, "adapters"), exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    tasks, _ = P["load_tasks"]()
    ex = type("X", (), {"log": staticmethod(log)})()
    states, _ = P["build_states"](tasks, ex)
    rng = random.Random(92)
    ids = sorted({s["iid"] for s in states})
    rng.shuffle(ids)
    n = len(ids)
    split = {i: ("fit" if k < n // 2 else "dev" if k < 3 * n // 4 else "test") for k, i in enumerate(ids)}
    held = [s for s in states if split[s["iid"]] in ("dev", "test")]
    quick = random.Random(99).sample(held, min(96, len(held)))
    sets = {"quick": quick, "held": held}

    def vprompt(prefix):
        p = P["G"]["render"](tok, prefix, True)
        return p[len("<bos>"):] if p.startswith("<bos>") else p
    prompts = {s["iid"]: vprompt(s["prefix"]) for s in held}
    json.dump({k: [s["iid"] for s in v] for k, v in sets.items()}, open(os.path.join(LAB, "splits.json"), "w"))
    log("states", len(states), "| quick", len(quick), "| held", len(held))
    md = P["official_model"]()
    # an engine left by an earlier job runs as its own process and keeps GPU memory (2026-11-09: the first lab start failed this way);
    # clear every vLLM process and wait until the GPU is free, as colab/remote/runjob.sh does
    subprocess.run("for p in $(ps -eo pid,args | grep -E 'vllm|VLLM|EngineCore' | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done", shell=True)
    for _ in range(60):
        used = int(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.split()[0])
        if used < 1500:
            break
        time.sleep(2)
    log("GPU memory in use before the lab's vLLM:", used, "MiB")
    cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--host", "127.0.0.1",
           "--port", str(PORT), "--max-model-len", "32768", "--gpu-memory-utilization", "0.85", "--enforce-eager", "--max-num-seqs", "192",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0}), "--enable-lora", "--max-lora-rank", "128", "--max-loras", "8",
           "--max-cpu-loras", "16"]
    env = dict(os.environ, VLLM_ALLOW_RUNTIME_LORA_UPDATING="True")
    srv = subprocess.Popen(cmd, stdout=open(os.path.join(LAB, "vllm.log"), "w"), stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while True:
        if srv.poll() is not None:
            raise RuntimeError("vLLM exited: " + open(os.path.join(LAB, "vllm.log"), errors="replace").read()[-12000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    log("vLLM ready in", round(time.time() - t0), "s; waiting for jobs in", os.path.join(LAB, "jobs"))
    open(os.path.join(LAB, "READY"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    idle = time.time()
    done_jobs = 0
    try:
        while not os.path.exists(os.path.join(LAB, "STOP")) and time.time() - idle < 6 * 3600:
            todo = [j for j in sorted(glob.glob(os.path.join(LAB, "jobs", "*.json")))
                    if not os.path.exists(os.path.join(LAB, "results", os.path.basename(j)))]
            if not todo:
                if os.environ.get("FASTLAB_ONCE") and done_jobs:  # run-once mode for an experiment that brings its own lab (EXP-102)
                    break
                time.sleep(10)
                continue
            jf = todo[0]
            job = json.load(open(jf))
            name = job["name"]
            ss = sets[job.get("split", "quick")]
            caps = tuple(job.get("caps") or CAPS[job.get("split", "quick")])
            t1 = time.time()
            loaded, errors = [], {}
            for arm, path in job["arms"].items():
                ln = f"{name}__{arm}"
                try:
                    out = req("/v1/load_lora_adapter", dict(lora_name=ln, lora_path=path), timeout=600)
                    loaded.append((arm, ln))
                except Exception as e:  # noqa: BLE001
                    errors[arm] = repr(e)[:300]
            # prompt variants: the unedited model with a text appended to the system message (a variant arm per entry)
            variants = job.get("variants") or {}
            sep = chr(10) * 2
            vprompts = {vn: {s["iid"]: vprompt([dict(m, content=m["content"] + sep + txt) if m.get("role") == "system" else m for m in s["prefix"]])
                             for s in ss} for vn, txt in variants.items()}
            models = [("base", "base")] + loaded + [(vn, "base") for vn in variants]
            work = [(arm, ln, s) for arm, ln in models for s in ss]

            def one(w):
                arm, ln, s = w
                try:
                    t = answer(ln, (vprompts[arm] if arm in vprompts else prompts)[s["iid"]], caps)
                except Exception as e:  # noqa: BLE001
                    return arm, s["iid"], dict(applies=False, correct=False, tool=None, esc=False, selfcopy=False, error=repr(e)[:120])
                sc = P["score"](t, s)
                return arm, s["iid"], dict(sc, esc=escape_error(t, s), selfcopy=self_copy(t, s), n_chars=len(t))
            with ThreadPoolExecutor(256) as pool:
                rows = list(pool.map(one, work))
            res = {arm: {} for arm, _ in models}
            for arm, iid, r in rows:
                res[arm][iid] = r
            for arm, ln in loaded:
                try:
                    req("/v1/unload_lora_adapter", dict(lora_name=ln), timeout=300)
                except Exception:  # noqa: BLE001
                    pass
            out = dict(name=name, split=job.get("split", "quick"), caps=caps, n=len(ss), order=[s["iid"] for s in ss], arms=res, load_errors=errors,
                       wall_s=round(time.time() - t1, 1))
            json.dump(out, open(os.path.join(LAB, "results", os.path.basename(jf)), "w"))
            summ = {a: (sum(r["applies"] for r in v.values()), sum(r["correct"] for r in v.values()), sum(r["esc"] for r in v.values())) for a, v in res.items()}
            log("job", name, "done in", out["wall_s"], "s | applies, correct, escapes:", summ, "| load errors:", errors)
            done_jobs += 1
            idle = time.time()
    finally:
        srv.terminate()
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()

"""EXP-112: candidate edits on the first-edit bench (FEB): real agent states, the decisive event.

On the 7 tasks E2B can solve, a run whose first edit applies is solved in 72% of cases (99 of 137), one whose first edit fails in
14% (10 of 71), one that never edits in 0% (tools/analysis/first_edit.py). FEB (tools/analysis/build_first_edit_bench.py) holds the
agent state right before E2B's first edit_file call in its official runs, with the edited file as it stood then, split by the
shared dev/test task halves. This run answers each test-half state once per arm with the official W4A16 E2B on vLLM (official
sampling: temperature 0.2, top_p 0.95, a fixed seed per state; thinking closed at 2,048 tokens, then up to 512 more), parses the
first tool call, and scores it with the official matcher (adk_eval_core.editing.edit.apply_replacement) against the file:
  applies (an edit_file call on the recorded file whose old_string the official matcher accepts), not_found, escape (old_string
  carries a backslash-quote the file lacks), other_file, other_tool, no_call.
Arms (one server, same states): unedited; a zero LoRA (noise reference); the compiled instruction at layer 10 and its neutral
control (EXP-104); escape-blind values at strength 2 and their control (EXP-099); the four 31B teacher neurons (EXP-080); the E4B
transplant (EXP-102); the combined edit and the combined control (EXP-111); and the instruction as text in the system message.
Paired exact McNemar on 'applies' against the unedited arm and each edit against its control.
"""
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

A = os.path.join(ROOT, "artifacts_local")
ARMS = {"zero": f"{A}/fastlab_zero/zero_v", "ins_L10": f"{A}/EXP-104/ins_L10", "ctl_L10": f"{A}/EXP-104/ctl_L10",
        "esc014_a2": f"{A}/EXP-099/esc014_a2", "ctrl014_a2": f"{A}/EXP-099/ctrl014_a2", "teacher": f"{A}/EXP-080/teacher_all",
        "e4b_tx": f"{A}/EXP-102/st_a1", "combo": f"{A}/EXP-111/combo_all", "combo_ctl": f"{A}/EXP-111/combo_control"}
PAIRS = [("ins_L10", "ctl_L10"), ("esc014_a2", "ctrl014_a2"), ("combo", "combo_ctl")]
INSTR = ("The bug is one small mistake on one line. Once you have read the code that contains it, fix it right away with edit_file. "
         "Copy old_string character for character from the file content you read, without adding escape characters, and in new_string "
         "change only the wrong token or operator. Do not rewrite or restructure code.")
PORT = 8019
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)


def official():
    base = os.path.join(ROOT, "tools", "harness", "src_adk_eval_core", "adk_eval_core")
    pkg = types.ModuleType("adk_eval_core")
    pkg.__path__ = [base]
    sys.modules.setdefault("adk_eval_core", pkg)
    mod = None
    for name, rel in [("adk_eval_core.errors", "errors.py"), ("official_edit", os.path.join("editing", "edit.py"))]:
        spec = importlib.util.spec_from_file_location(name, os.path.join(base, rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def mcnemar(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def req(path, body, timeout=3600):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def with_instruction(prompt):
    k = prompt.find("<turn|>")  # end of the system turn
    return prompt if k < 0 else prompt[:k] + chr(10) * 2 + INSTR + prompt[k:]


with Experiment("EXP-112", "first-edit-bench", category="improvement", depends_on=["EXP-104", "EXP-099", "EXP-080", "EXP-102", "EXP-111"],
                motivation="The first edit applying decides success on solvable tasks (72% against 14%); the synthetic bench did not predict the "
                           "agent, so candidate edits are screened on the agent's own first-edit states.",
                hypothesis="An edit raises the share of first edits that apply on held-out real states above the unedited model and above its "
                           "matched control.",
                config=dict(arms=sorted(ARMS) + ["instruction_text"], caps=[2048, 512], sampling=dict(temperature=0.2, top_p=0.95))) as ex:
    OFF = official()
    # none of the edits here was fitted on FEB states, so both halves are used (461 states); FEB's dev half also holds all 7 tasks
    # E2B can solve, reported apart (D-121)
    SOLVABLE = {"fastapi_14786", "rich_2943", "requests_6644", "fastapi_14301", "rich_3006", "rich_3454", "fastapi_9425"}
    states = [dict(json.loads(l), half=h) for h in ("dev", "test") for l in open(os.path.join(A, "FEB", f"feb_{h}.jsonl"), encoding="utf-8")]
    if SMOKE:
        states = states[:6]
    ex.log("states", len(states), "| tasks", len({s["iid"] for s in states}), "| originally applied", sum(s["orig"] == "applied" for s in states))
    md = next(p for p in __import__("glob").glob("/content/kaggle_input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/*") if os.path.exists(os.path.join(p, "config.json")))
    arms = {k: v for k, v in ARMS.items() if os.path.exists(os.path.join(v, "adapter_model.safetensors"))}
    ex.log("adapters found", sorted(arms))
    cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--host", "127.0.0.1",
           "--port", str(PORT), "--max-model-len", "32768", "--gpu-memory-utilization", "0.88", "--enforce-eager", "--max-num-seqs", "128",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0}), "--enable-lora", "--max-lora-rank", "128", "--max-loras", str(max(1, len(arms))),
           "--max-cpu-loras", str(max(1, len(arms))), "--lora-modules", *[f"{k}={v}" for k, v in arms.items()]]
    log = os.path.join(ex.dir, "vllm.log")
    srv = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        if srv.poll() is not None:
            raise RuntimeError("vLLM exited: " + open(log, errors="replace").read()[-4000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    ex.log("vLLM ready in", round(time.time() - t0), "s")

    def answer(model, prompt, seed):
        r = req("/v1/completions", dict(model=model, prompt=prompt, max_tokens=2048, temperature=0.2, top_p=0.95, seed=seed, skip_special_tokens=False,
                                        stop=["<tool_call|>"]))
        t = r["choices"][0]["text"]
        if r["choices"][0].get("finish_reason") == "length" and "<channel|>" not in t and "<|tool_call>" not in t:
            r2 = req("/v1/completions", dict(model=model, prompt=prompt + t + "<channel|>", max_tokens=512, temperature=0.2, top_p=0.95, seed=seed,
                                             skip_special_tokens=False, stop=["<tool_call|>"]))
            t = t + "<channel|>" + r2["choices"][0]["text"]
        return t

    def score(text, st):
        m = CALL.search(text or "")
        if not m:
            return "no_call"
        if m.group(1) != "edit_file":
            return "other_tool"
        a = dict(ARG.findall(m.group(2)))
        fp = (a.get("filepath") or "").removeprefix("/workspace/").lstrip("./")
        if fp != (st["filepath"] or "").removeprefix("/workspace/").lstrip("./"):
            return "other_file"
        old, new = a.get("old_string") or "", a.get("new_string") or ""
        if not old:
            return "empty"
        r = OFF.apply_replacement(st["content"], old, new, allow_multiple=False)
        if not r.error_message:
            return "applies"
        if '\\"' in old and '\\"' not in st["content"]:
            return "escape"
        return "multiple" if r.occurrences > 1 else "not_found"
    models = [("base", "base", False)] + [(k, k, False) for k in arms] + [("instruction_text", "base", True)]
    work = [(arm, mdl, instr, k, s) for arm, mdl, instr in models for k, s in enumerate(states)]

    def one(w):
        arm, mdl, instr, k, s = w
        try:
            t = answer(mdl, with_instruction(s["prompt"]) if instr else s["prompt"], 1000 + k)
            return arm, s["id"], score(t, s)
        except Exception as e:  # noqa: BLE001
            return arm, s["id"], "error:" + repr(e)[:80]
    t1 = time.time()
    with ThreadPoolExecutor(128) as pool:
        rows = list(pool.map(one, work))
    ex.log("answers", len(rows), "in", round(time.time() - t1), "s")
    srv.terminate()
    res = {}
    for arm, sid, outcome in rows:
        res.setdefault(arm, {})[sid] = outcome
    ex.save_json("rows.json", res)
    ids = [s["id"] for s in states]
    base = res["base"]
    summ = {}
    for arm, v in res.items():
        c = {}
        for o in v.values():
            k_ = o.split(":")[0]
            c[k_] = c.get(k_, 0) + 1
        solv = [s["id"] for s in states if s["iid"] in SOLVABLE]
        row = dict(counts=c, applies=sum(v[i] == "applies" for i in ids), applies_solvable_tasks=sum(v[i] == "applies" for i in solv), solvable_states=len(solv))
        if arm != "base":
            x = sum(1 for i in ids if v[i] == "applies" and base[i] != "applies")
            y = sum(1 for i in ids if base[i] == "applies" and v[i] != "applies")
            row["vs_base"] = (x, y, round(mcnemar(x, y), 4))
        summ[arm] = row
        ex.log(arm, row)
    for e, c in PAIRS:
        if e in res and c in res:
            x = sum(1 for i in ids if res[e][i] == "applies" and res[c][i] != "applies")
            y = sum(1 for i in ids if res[c][i] == "applies" and res[e][i] != "applies")
            summ[f"{e}_vs_{c}"] = (x, y, round(mcnemar(x, y), 4))
            ex.log(f"{e} vs {c}", summ[f"{e}_vs_{c}"])
    ex.result(states=len(states), summary=summ, headline="first-edit bench, edits that apply of " + str(len(states)) + ": " +
              ", ".join(f"{a} {summ[a]['applies']}" for a in summ if isinstance(summ[a], dict)))

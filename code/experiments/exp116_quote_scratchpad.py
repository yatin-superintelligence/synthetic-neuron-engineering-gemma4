"""EXP-116: is E2B's thought a copy scratchpad? Forced quoting, a neutral prefix and thinking closed, on the first-edit bench (FEB).

In the official runs E2B's thought quotes code in a fenced block before 76.9% of first edits that apply and 43.7% of those that
fail; E4B quotes before about 90% of both (tools/analysis/thought_quote.py). This run tests the link causally on FEB's 461 real
states (the agent state right before E2B's first edit_file call), with the official W4A16 E2B on vLLM and official sampling
(temperature 0.2, top_p 0.95, a fixed seed per state; thinking closed at 2,048 tokens, then up to 512 more), arms:
  base     the state as recorded (the thought starts empty);
  quote    the thought starts with a sentence committing to copy the lines to change, then an opened python code block;
  neutral  the thought starts with a sentence of similar length that commits to restating the issue (no code, no copying);
  fence    the thought starts with an opened python code block only (what one decision neuron at the first token can write);
  closed   the thought is closed at once (thinking off at this step).
Scored with the official matcher (adk_eval_core.editing.edit.apply_replacement) as in EXP-112, plus: whether the edited file then
holds the reference fix's change for that file (every reference-added line present, every reference-removed line gone;
artifacts_local/FEB/feb_ref.json, states whose file the reference touches), and whether the thought quoted code.
Paired exact McNemar on 'applies' and on 'holds the reference change' for quote against neutral and each arm against base.
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
PREFIX = {
    "base": "",
    "quote": "Before editing, I will copy the exact lines I am going to change from the file content above, character for character:\n```python\n",
    "neutral": "Before editing, I will restate in my own words what the issue asks for and which behaviour of the code is wrong.\n",
    "fence": "```python\n",  # only the opened code block: what a single decision neuron at the thought's first token can write
    "closed": None,
}
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


with Experiment("EXP-116", "quote-scratchpad", category="diagnosis", depends_on=["EXP-112"],
                motivation="E2B's thought quotes the code before 77% of first edits that apply and 44% of those that fail; whether quoting "
                           "causes the edit to apply decides which neuron (or setting) to build next.",
                hypothesis="Forcing the thought to open with a quote of the lines to change raises first edits that apply on real states above "
                           "a neutral prefix of similar length and above the recorded state.",
                config=dict(arms=list(PREFIX), caps=[2048, 512], sampling=dict(temperature=0.2, top_p=0.95), prefixes=PREFIX)) as ex:
    OFF = official()
    SOLVABLE = {"fastapi_14786", "rich_2943", "requests_6644", "fastapi_14301", "rich_3006", "rich_3454", "fastapi_9425"}
    states = [dict(json.loads(l), half=h) for h in ("dev", "test") for l in open(os.path.join(A, "FEB", f"feb_{h}.jsonl"), encoding="utf-8")]
    REF = json.load(open(os.path.join(A, "FEB", "feb_ref.json"), encoding="utf-8"))
    if SMOKE:
        states = states[:6]
    # 39 of 461 FEB states sit at a new model turn, where the template stops at the turn opener and E2B writes the thought opener
    # itself; it is appended so every arm starts from an opened thought
    for s in states:
        if s["prompt"].endswith("<|turn>model\n"):
            s["prompt"] += "<|channel>thought\n"
    assert all(s["prompt"].endswith("<|channel>thought\n") for s in states), "FEB prompts must end with the opened thought"
    ex.log("states", len(states), "| with a reference change in their file", sum(1 for s in states if REF[s["id"]]["plus"] or REF[s["id"]]["minus"]))
    md = next(p for p in __import__("glob").glob("/content/kaggle_input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/*") if os.path.exists(os.path.join(p, "config.json")))
    cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--host", "127.0.0.1",
           "--port", str(PORT), "--max-model-len", "32768", "--gpu-memory-utilization", "0.88", "--enforce-eager", "--max-num-seqs", "128",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]
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

    def gen(prompt, n, seed):
        r = req("/v1/completions", dict(model="base", prompt=prompt, max_tokens=n, temperature=0.2, top_p=0.95, seed=seed, skip_special_tokens=False,
                                        stop=["<tool_call|>"]))
        return r["choices"][0]["text"], r["choices"][0].get("finish_reason")

    def answer(arm, prompt, seed):
        pre = PREFIX[arm]
        if pre is None:  # thought closed at once
            t, _ = gen(prompt + "<channel|>", 1024, seed)
            return "<channel|>" + t
        t, fin = gen(prompt + pre, 2048, seed)
        t = pre + t
        if fin == "length" and "<channel|>" not in t and "<|tool_call>" not in t:
            t2, _ = gen(prompt + t + "<channel|>", 512, seed)
            t = t + "<channel|>" + t2
        return t

    def holds_ref(new, st):
        rf = REF[st["id"]]
        if not (rf["plus"] or rf["minus"]):
            return None
        have = {l.strip() for l in new.splitlines()}
        before = {l.strip() for l in st["content"].splitlines()}
        return all(p in have for p in rf["plus"]) and all(m not in have for m in rf["minus"] if m in before and m not in rf["plus"])

    def score(text, st):
        thought = text.split("<channel|>")[0]
        quoted = "```" in thought
        m = CALL.search(text or "")
        if not m:
            return dict(o="no_call", quoted=quoted, ref=None)
        if m.group(1) != "edit_file":
            return dict(o="other_tool", quoted=quoted, ref=None)
        a = dict(ARG.findall(m.group(2)))
        fp = (a.get("filepath") or "").removeprefix("/workspace/").lstrip("./")
        if fp != (st["filepath"] or "").removeprefix("/workspace/").lstrip("./"):
            return dict(o="other_file", quoted=quoted, ref=None)
        old, new = a.get("old_string") or "", a.get("new_string") or ""
        if not old:
            return dict(o="empty", quoted=quoted, ref=None)
        in_thought = any(l.strip() in thought for l in old.splitlines() if len(l.strip()) >= 12)
        r = OFF.apply_replacement(st["content"], old, new, allow_multiple=False)
        if not r.error_message:
            return dict(o="applies", quoted=quoted, ref=holds_ref(r.new_content, st), old_in_thought=in_thought)
        if old == new:
            return dict(o="noop", quoted=quoted, ref=None, old_in_thought=in_thought)
        if '\\"' in old and '\\"' not in st["content"]:
            return dict(o="escape", quoted=quoted, ref=None, old_in_thought=in_thought)
        return dict(o="multiple" if r.occurrences > 1 else "not_found", quoted=quoted, ref=None, old_in_thought=in_thought)

    work = [(arm, k, s) for arm in PREFIX for k, s in enumerate(states)]

    def one(w):
        arm, k, s = w
        try:
            return arm, s["id"], score(answer(arm, s["prompt"], 1000 + k), s)
        except Exception as e:  # noqa: BLE001
            return arm, s["id"], dict(o="error:" + repr(e)[:80], quoted=False, ref=None)
    t1 = time.time()
    with ThreadPoolExecutor(128) as pool:
        rows = list(pool.map(one, work))
    ex.log("answers", len(rows), "in", round(time.time() - t1), "s")
    srv.terminate()
    res = {}
    for arm, sid, r in rows:
        res.setdefault(arm, {})[sid] = r
    ex.save_json("rows.json", res)
    ids = [s["id"] for s in states]
    solv = [s["id"] for s in states if s["iid"] in SOLVABLE]
    refd = [i for i in ids if REF[i]["plus"] or REF[i]["minus"]]
    summ = {}
    for arm, v in res.items():
        c = {}
        for r in v.values():
            k_ = r["o"].split(":")[0]
            c[k_] = c.get(k_, 0) + 1
        row = dict(counts=c, applies=sum(v[i]["o"] == "applies" for i in ids), holds_reference=sum(bool(v[i]["ref"]) for i in refd),
                   reference_states=len(refd), quoted=sum(v[i]["quoted"] for i in ids),
                   applies_solvable_tasks=sum(v[i]["o"] == "applies" for i in solv), holds_reference_solvable=sum(bool(v[i]["ref"]) for i in solv if i in refd),
                   solvable_states=len(solv))
        if arm != "base":
            for key, f in (("applies", lambda r: r["o"] == "applies"), ("holds_reference", lambda r: bool(r["ref"]))):
                pool_ = ids if key == "applies" else refd
                x = sum(1 for i in pool_ if f(v[i]) and not f(res["base"][i]))
                y = sum(1 for i in pool_ if f(res["base"][i]) and not f(v[i]))
                row[f"{key}_vs_base"] = (x, y, round(mcnemar(x, y), 4))
        summ[arm] = row
        ex.log(arm, row)
    for key, f, pool_ in (("applies", lambda r: r["o"] == "applies", ids), ("holds_reference", lambda r: bool(r["ref"]), refd)):
        x = sum(1 for i in pool_ if f(res["quote"][i]) and not f(res["neutral"][i]))
        y = sum(1 for i in pool_ if f(res["neutral"][i]) and not f(res["quote"][i]))
        summ[f"quote_vs_neutral_{key}"] = (x, y, round(mcnemar(x, y), 4))
        ex.log(f"quote vs neutral, {key}", summ[f"quote_vs_neutral_{key}"])
    # observational check inside the base arm: applied share when the thought quoted code against when it did not
    b = res["base"]
    q = [i for i in ids if b[i]["quoted"] and b[i]["o"] not in ("no_call", "other_tool")]
    nq = [i for i in ids if not b[i]["quoted"] and b[i]["o"] not in ("no_call", "other_tool")]
    summ["base_edit_calls_applied_when_quoted"] = (sum(b[i]["o"] == "applies" for i in q), len(q))
    summ["base_edit_calls_applied_when_not_quoted"] = (sum(b[i]["o"] == "applies" for i in nq), len(nq))
    ex.result(states=len(states), summary=summ, headline="FEB, first edits that apply of " + str(len(states)) + " / hold the reference change of " +
              str(len(refd)) + ": " + ", ".join(f"{a} {summ[a]['applies']}/{summ[a]['holds_reference']}" for a in PREFIX))

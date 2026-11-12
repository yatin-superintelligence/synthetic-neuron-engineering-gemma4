"""EXP-124 (laptop, free): does the E4B transplant make E2B's first edits more correct on real agent states? F16 llama.cpp on the Arc GPU.

Why: E2B's bottleneck is writing the correct fix (first edits that leave the reference change: about 19 to 25 of 281 FEB states in
every arm so far, D-133, D-141). The E4B transplant (EXP-102's st_a1: 896 E4B neurons translated into 7 E2B layers) is the only edit
with a replicated capability gain (reference-fix token accuracy +1.1 points, p = 2e-6), but its effect on correct edits at real
states was never measured. This run measures it for free on the laptop while Colab runs EXP-109e and EXP-123.
States: FEB states whose file the reference fix touches (artifacts_local/FEB/feb_ref.json), prompts of at most 12,000 tokens.
Arms (one llama-server, F16 GGUF, one slot, official sampling: temperature 0.2, top_p 0.95, seed 1000 + index; thinking forced closed
at 4,096 tokens as vLLM does): unedited and the transplant (scale 1). Per state the arms run back to back so the prompt cache is reused.
Measures: first edits that apply (official matcher), edits that leave the reference change; exact McNemar on both. Rows are appended to
rows.jsonl as they come, so an interrupted run keeps its results.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, loops as L  # noqa: E402
from sne.artifacts import LOCAL  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402
from sne.stats import mcnemar_exact  # noqa: E402

A = os.path.join(ROOT, "artifacts_local")
SRC = os.path.join(A, "EXP-102", "st_a1")
MAX_PROMPT = 12000
OPENER = "<|channel>thought\n"
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)


def build(exp_dir):
    """Child process: a GGUF LoRA from EXP-102's PEFT adapter (gate, up and down projections of 7 layers, rank 128, alpha = r)."""
    from safetensors.torch import load_file
    from sne.artifacts import build_adapters
    ex = SimpleNamespace(id="EXP-124", dir=exp_dir)
    sd = load_file(os.path.join(SRC, "adapter_model.safetensors"))
    cfg = json.load(open(os.path.join(SRC, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]

    def fn(e, bw, prec):
        keys = sorted(k for k in sd if k.endswith("lora_A.weight"))
        for k in keys:
            i = int(k.split(".layers.")[1].split(".")[0])
            mod = "mlp." + k.split(".mlp.")[1].split(".lora_A")[0]
            e.add(i, mod, sd[k.replace("lora_A", "lora_B")].float() * scale, sd[k].float())
        e.meta.update(source="EXP-102/st_a1", modules=len(keys))
    build_adapters(ex, "e4b_tx", fn, precisions=("f16",))


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


if "--build" in sys.argv:
    build(sys.argv[sys.argv.index("--build") + 1])
    sys.exit(0)

with Experiment("EXP-124", "local-transplant-correctness", category="improvement", depends_on=["EXP-102", "EXP-112", "EXP-116"],
                motivation="E2B fails mainly at writing the correct fix; the E4B transplant is the one edit with a replicated capability gain, "
                           "and its effect on correct first edits at real states was never measured.",
                hypothesis="On FEB states whose file the reference touches, the transplant raises first edits that leave the reference change "
                           "above unedited (exact McNemar).",
                config=dict(adapter="EXP-102/st_a1", precision="f16", max_prompt=MAX_PROMPT, sampling=dict(temperature=0.2, top_p=0.95))) as ex:
    OFF = official()
    tok = tokenizer()
    REF = json.load(open(os.path.join(A, "FEB", "feb_ref.json"), encoding="utf-8"))
    states = [json.loads(l) for h in ("dev", "test") for l in open(os.path.join(A, "FEB", f"feb_{h}.jsonl"), encoding="utf-8")]
    states = [s for s in states if REF[s["id"]]["plus"] or REF[s["id"]]["minus"]]
    for s in states:
        if s["prompt"].endswith("<|turn>model\n"):
            s["prompt"] += OPENER
        s["n"] = len(tok(s["prompt"], add_special_tokens=False)["input_ids"])
    states = [s for s in states if s["n"] <= MAX_PROMPT]
    states.sort(key=lambda s: s["id"])
    if SMOKE:
        states = sorted(states, key=lambda s: s["n"])[:2]
    ex.log("states", len(states), "| tasks", len({s["iid"] for s in states}))
    subprocess.run([sys.executable, os.path.abspath(__file__), "--build", ex.dir], check=True)
    adapter = os.path.join(LOCAL, "EXP-124", "e4b_tx_f16.gguf")

    def holds_ref(new, st):
        rf = REF[st["id"]]
        have = {l.strip() for l in new.splitlines()}
        before = {l.strip() for l in st["content"].splitlines()}
        return all(p in have for p in rf["plus"]) and all(m not in have for m in rf["minus"] if m in before and m not in rf["plus"])

    def score(text, st):
        m = CALL.search(text or "")
        if not m:
            return dict(o="no_call", ref=False)
        if m.group(1) != "edit_file":
            return dict(o="other_tool", ref=False)
        a = dict(ARG.findall(m.group(2)))
        fp = (a.get("filepath") or "").removeprefix("/workspace/").lstrip("./")
        if fp != (st["filepath"] or "").removeprefix("/workspace/").lstrip("./"):
            return dict(o="other_file", ref=False)
        old, new = a.get("old_string") or "", a.get("new_string") or ""
        if not old:
            return dict(o="empty", ref=False)
        r = OFF.apply_replacement(st["content"], old, new, allow_multiple=False)
        if not r.error_message:
            return dict(o="applies", ref=holds_ref(r.new_content, st))
        return dict(o="noop" if old == new else ("multiple" if r.occurrences > 1 else "not_found"), ref=False)

    out_rows = os.path.join(ex.dir, "rows.jsonl")
    done = set()
    if os.path.exists(out_rows):
        done = {(r["id"], r["arm"]) for r in (json.loads(l) for l in open(out_rows, encoding="utf-8"))}
    with LlamaServer(GGUF["f16"], loras=[adapter], parallel=1, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                     log_path=os.path.join(ex.dir, "server_f16.log")) as srv:
        for k, s in enumerate(states):
            for arm, scales in (("base", {}), ("e4b_tx", {0: 1.0})):
                if (s["id"], arm) in done:
                    continue
                sampling = dict(temperature=0.2, top_p=0.95, top_k=0, seed=1000 + k, stop=L.STOP, cache_prompt=True)
                g = L.generate(srv, s["prompt"], s["n"], sampling, scales, 128 if SMOKE else L.THINK_BUDGET)
                row = dict(id=s["id"], iid=s["iid"], arm=arm, n_gen=g["n_gen"], **score(g["text"], s))
                with open(out_rows, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            if k % 10 == 0:
                ex.log("states done", k + 1, "of", len(states))
    rows = [json.loads(l) for l in open(out_rows, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["id"]] = r
    ids = sorted(set(by.get("base", {})) & set(by.get("e4b_tx", {})))
    summ = {}
    for arm in ("base", "e4b_tx"):
        v = by[arm]
        c = {}
        for i in ids:
            c[v[i]["o"]] = c.get(v[i]["o"], 0) + 1
        summ[arm] = dict(counts=c, applies=sum(v[i]["o"] == "applies" for i in ids), holds_reference=sum(bool(v[i]["ref"]) for i in ids))
    tests = {}
    for key, f in (("applies", lambda r: r["o"] == "applies"), ("holds_reference", lambda r: bool(r["ref"]))):
        x = sum(1 for i in ids if f(by["e4b_tx"][i]) and not f(by["base"][i]))
        y = sum(1 for i in ids if f(by["base"][i]) and not f(by["e4b_tx"][i]))
        tests[key] = dict(transplant_only=x, base_only=y, p=round(mcnemar_exact(x, y), 4))
    ex.result(states=len(ids), summary=summ, tests=tests,
              headline=f"F16 FEB, {len(ids)} states whose file the reference touches: applies base {summ['base']['applies']}, transplant "
                       f"{summ['e4b_tx']['applies']}; reference change held base {summ['base']['holds_reference']}, transplant {summ['e4b_tx']['holds_reference']}")

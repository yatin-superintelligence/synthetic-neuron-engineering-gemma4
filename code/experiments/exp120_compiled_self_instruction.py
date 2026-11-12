"""EXP-120: the forced self-instruction compiled into E2B's weights by state injection (closed-form distillation, EXP-104's solve).

Why: opening E2B's thought with "Before editing, I will copy the exact lines I am going to change from the file content above,
character for character:" and a code fence raised first edits that apply from 153 to 221 of 461 real states (D-133); logit-writing
neurons cannot write the sentence (D-137) and E2B ignores it as a system-prompt instruction (1 of 20 edit thoughts, D-138). This
experiment moves the effect of the sentence, not the sentence: the teacher is E2B with the sentence forced after its thought opener
and its own continuation; the student is the same state without the sentence, followed by the same continuation. At aligned
positions (the student's thought start and its continuation, which the teacher reaches one sentence later) the change of layer L's
output d = h(teacher) - h(student) is fitted as a closed-form down_proj edit; shared prompt positions keep d = 0.
Steps (one L4 job): (1) vLLM, unedited: the continuation after the forced sentence (and after the neutral sentence, for the control)
for dev thought starts after a read or a code search (official sampling, seed 120 + index); (2) bf16 fit in a child process, layers 10
and 14 (quote) and 10 (neutral control), rank 128 each; (3) vLLM with the adapters on FEB's 461 states and test-half reads not followed
by an edit (premature edits): base, inj_q_L10, inj_q_L14, inj_n_L10 and the forced quote prefix itself for reference.
"""
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

A = os.path.join(ROOT, "artifacts_local")
EXP = "EXP-120"
OUT = os.path.join(A, EXP)
FIT_OUT = f"/content/{EXP}_fit.json"
CONT_FILE = f"/content/{EXP}_continuations.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SENT = {
    "q": "Before editing, I will copy the exact lines I am going to change from the file content above, character for character:\n```python\n",
    "n": "Before editing, I will restate in my own words what the issue asks for and which behaviour of the code is wrong.\n",
}
FITS = [("q", 10), ("q", 14), ("n", 10)]
RANK = 128
N_CONT = 384  # continuation tokens used per state
N_ZERO = 64
MAX_TOKENS = 12000
OPENER = "<|channel>thought\n"
PORT = 8025
TRIGGER_PREV = ("read_file", "search_similar_code")
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)


def inv_scale(w):
    e = (0.05 * w.abs().median()) ** 2
    return w / (w * w + e)


def save_lora(path, L, A_, B_):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    base = f"base_model.model.model.language_model.layers.{L}.mlp.down_proj"
    save_file({f"{base}.lora_A.weight": A_.to(torch.bfloat16).contiguous(), f"{base}.lora_B.weight": B_.to(torch.bfloat16).contiguous()},
              os.path.join(path, "adapter_model.safetensors"))
    r = A_.shape[0]
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=r, lora_alpha=r, target_modules=["down_proj"],
                   layers_to_transform=[L], lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
                   use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


def fit_states():
    rows = [json.loads(l) for l in open(os.path.join(A, "EXP-117", "thought_starts_dev.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["prev"] in TRIGGER_PREV]
    return rows[:4] if SMOKE else rows


def fit_stage():
    import gc
    from huggingface_hub import snapshot_download
    from sne.model import layers, load
    log = lambda *a: print(time.strftime("[%H:%M:%S]"), *a, flush=True)  # noqa: E731
    cont = json.load(open(os.environ.get("EXP120_CONT") or CONT_FILE, encoding="utf-8"))
    rows = [r for r in fit_states() if r["id"] in cont["q"]]
    model, tok = load(os.environ.get("EXP120_MODEL") or snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat"),
                      ple_on_cpu=False, dev=DEV, **({"dtype": torch.float32} if os.environ.get("EXP120_FP32") else {}))
    Ls = layers(model)
    LAYS = sorted({L for _, L in FITS})
    st = {}

    def keep_down(m, a, o, L):  # returns None (a hook that returns a value replaces the module's output)
        st[("k", L)] = a[0][0]
        st[("y", L)] = o[0]

    def keep_layer(m, a, o, L):
        st[("h", L)] = (o[0] if isinstance(o, tuple) else o)[0]
    hooks = []
    for L in LAYS:
        hooks.append(Ls[L].mlp.down_proj.register_forward_hook(lambda m, a, o, L=L: keep_down(m, a, o, L)))
        hooks.append(Ls[L].register_forward_hook(lambda m, a, o, L=L: keep_layer(m, a, o, L)))
    rng = random.Random(120)
    rec = {(t, L): dict(K=[], Y=[], D=[]) for t, L in FITS}
    used = {"q": 0, "n": 0}
    with torch.no_grad():
        for r in rows:
            P_ids = [tok.bos_token_id] + tok(r["prompt"], add_special_tokens=False)["input_ids"]
            if len(P_ids) + 600 > MAX_TOKENS:
                continue
            for tag in ("q", "n"):
                if not any(t == tag for t, _ in FITS) or r["id"] not in cont[tag]:
                    continue
                S_ids = tok(SENT[tag], add_special_tokens=False)["input_ids"]
                C_ids = tok(cont[tag][r["id"]], add_special_tokens=False)["input_ids"][:N_CONT]
                if len(C_ids) < 8:
                    continue
                student = P_ids + C_ids
                teacher = P_ids + S_ids + C_ids
                t0 = len(P_ids) - 1  # the student's thought start (it predicts the first continuation token)
                al_s = torch.tensor(list(range(t0, len(student) - 1)), device=DEV)  # positions predicting continuation tokens
                al_t = al_s + len(S_ids)
                zero = torch.tensor(sorted(rng.sample(range(1, t0 - 3), min(N_ZERO, t0 - 4))), device=DEV)
                model(input_ids=torch.tensor([student], device=DEV), use_cache=False, logits_to_keep=1)
                S_rec = {L: (st[("k", L)][torch.cat([al_s, zero])].half().cpu(), st[("y", L)][torch.cat([al_s, zero])].float().pow(2).mean(-1).sqrt().cpu(),
                             st[("h", L)][al_s].float().cpu()) for L in LAYS}
                st.clear()
                model(input_ids=torch.tensor([teacher], device=DEV), use_cache=False, logits_to_keep=1)
                for t, L in FITS:
                    if t != tag:
                        continue
                    d = st[("h", L)][al_t].float().cpu() - S_rec[L][2]
                    rec[(t, L)]["D"].append(torch.cat([d, torch.zeros(len(zero), d.shape[1])]).half())
                    rec[(t, L)]["K"].append(S_rec[L][0])
                    rec[(t, L)]["Y"].append(S_rec[L][1])
                st.clear()
                used[tag] += 1
            if (used["q"] + used["n"]) % 40 == 0:
                if DEV == "cuda":
                    torch.cuda.empty_cache()
                log(f"[inject] states used {used}")
    for h in hooks:
        h.remove()
    W = {L: dict(wp=Ls[L].post_feedforward_layernorm.weight.float().cpu(), s=float(Ls[L].layer_scalar.float()), Wn=float(Ls[L].mlp.down_proj.weight.float().norm()))
         for L in LAYS}
    del model
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    rep = dict(states_used=used, fits={})
    arms = {}
    for t, L in FITS:
        R = rec[(t, L)]
        if not R["K"]:
            continue
        K = torch.cat(R["K"]).double().to(DEV)
        rms = torch.cat(R["Y"]).double().to(DEV)
        D = torch.cat(R["D"]).double().to(DEV)
        G = K.T @ K
        G += 1e-2 * float(torch.diagonal(G).mean()) * torch.eye(G.shape[0], device=DEV, dtype=G.dtype)
        Gc = torch.linalg.cholesky(G)
        dY = (D / W[L]["s"]) * inv_scale(W[L]["wp"]).double().to(DEV)[None, :] * rms[:, None]
        dW = torch.cholesky_solve(K.T @ dY, Gc).T
        fitted = K @ dW.T
        r2 = float(1 - ((fitted - dY) ** 2).sum() / ((dY - dY.mean(0)) ** 2).sum())
        U, S, Vh = torch.linalg.svd(dW.float(), full_matrices=False)
        r = min(RANK, len(S))
        name = f"inj_{t}_L{L}"
        rep["fits"][name] = dict(rows=len(K), target_norm_mean=round(float(D.norm(dim=1).mean()), 4), fit_r2=round(r2, 4),
                                 rank_energy_share=round(float(S[:r].pow(2).sum() / S.pow(2).sum()), 4), delta_rel_to_W=round(float(dW.norm()) / W[L]["Wn"], 5))
        log(name, rep["fits"][name])
        arms[name] = save_lora(os.path.join(OUT, name), L, Vh[:r].cpu(), (U[:, :r] * S[:r]).cpu())
        del K, rms, D, G, Gc, dY, dW, fitted
        if DEV == "cuda":
            torch.cuda.empty_cache()
    rep["arms"] = arms
    json.dump(rep, open(FIT_OUT, "w"))


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


def start_server(arms, log_path):
    md = next(p for p in __import__("glob").glob("/content/kaggle_input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/*") if os.path.exists(os.path.join(p, "config.json")))
    cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--host", "127.0.0.1",
           "--port", str(PORT), "--max-model-len", "32768", "--gpu-memory-utilization", "0.88", "--enforce-eager", "--max-num-seqs", "128",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]
    if arms:
        cmd += ["--enable-lora", "--max-lora-rank", str(RANK), "--max-loras", str(len(arms)), "--max-cpu-loras", str(len(arms)),
                "--lora-modules", *[f"{k}={v}" for k, v in arms.items()]]
    srv = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    while True:
        if srv.poll() is not None:
            raise RuntimeError("vLLM exited: " + open(log_path, errors="replace").read()[-4000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            return srv
        except Exception:  # noqa: BLE001
            time.sleep(5)


def req(body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as f:
        return json.loads(f.read())


def answer(model, prompt, seed, prefix=""):
    r = req(dict(model=model, prompt=prompt + prefix, max_tokens=2048, temperature=0.2, top_p=0.95, seed=seed, skip_special_tokens=False, stop=["<tool_call|>"]))
    t = prefix + r["choices"][0]["text"]
    if r["choices"][0].get("finish_reason") == "length" and "<channel|>" not in t and "<|tool_call>" not in t:
        r2 = req(dict(model=model, prompt=prompt + t + "<channel|>", max_tokens=512, temperature=0.2, top_p=0.95, seed=seed, skip_special_tokens=False,
                      stop=["<tool_call|>"]))
        t = t + "<channel|>" + r2["choices"][0]["text"]
    return t


if __name__ == "__main__" and os.environ.get("EXP120_STAGE") == "fit":
    fit_stage()
elif __name__ == "__main__":
    with Experiment(EXP, "compiled-self-instruction", category="improvement", depends_on=["EXP-104", "EXP-116", "EXP-118", "EXP-119"],
                    motivation="The forced self-instruction raises applying first edits by 44%; neurons cannot write it and E2B ignores it as an "
                               "instruction, so its effect on E2B's states is compiled into a mid layer instead.",
                    hypothesis="The compiled prefix raises first edits that apply on FEB above unedited and above the compiled neutral sentence.",
                    config=dict(fits=FITS, rank=RANK, continuation_tokens=N_CONT, sentences=SENT)) as ex:
        os.makedirs(OUT, exist_ok=True)
        # (1) teacher continuations, unedited model
        srv = start_server({}, os.path.join(ex.dir, "vllm_gen.log"))
        rows = fit_states()
        cont = {"q": {}, "n": {}}

        def gen(w):
            tag, k, r = w
            try:
                t = answer("base", r["prompt"], 120 + k, prefix=SENT[tag])
                return tag, r["id"], t[len(SENT[tag]):]
            except Exception as e:  # noqa: BLE001
                return tag, r["id"], None
        with ThreadPoolExecutor(128) as pool:
            for tag, rid, t in pool.map(gen, [(tag, k, r) for tag in ("q", "n") for k, r in enumerate(rows)]):
                if t:
                    cont[tag][rid] = t
        srv.terminate()
        srv.wait()
        json.dump(cont, open(CONT_FILE, "w"))
        ex.log("continuations", {k: len(v) for k, v in cont.items()}, "| quote continuations whose thought quotes code:",
               sum("```" in t.split("<channel|>")[0] for t in cont["q"].values()))
        # (2) the fit, in a child process (vLLM needs the GPU memory back afterwards)
        t0 = time.time()
        flog = f"/content/{EXP}_fit.log"
        with open(flog, "w") as fl:
            fit = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, EXP120_STAGE="fit"), stdout=fl, stderr=subprocess.STDOUT)
        ex.log("fit stage exit", fit.returncode, "in", round(time.time() - t0), "s; log tail:", open(flog).read()[-3000:])
        rep = json.load(open(FIT_OUT))
        ex.save_json("fit.json", rep)
        # (3) evaluation
        OFF = official()
        REF = json.load(open(os.path.join(A, "FEB", "feb_ref.json"), encoding="utf-8"))
        feb = [dict(json.loads(l), kind="feb") for h in ("dev", "test") for l in open(os.path.join(A, "FEB", f"feb_{h}.jsonl"), encoding="utf-8")]
        for s in feb:
            if s["prompt"].endswith("<|turn>model\n"):
                s["prompt"] += OPENER
        prem = [dict(json.loads(l), kind="premature") for l in open(os.path.join(A, "EXP-117", "thought_starts_test.jsonl"), encoding="utf-8")]
        prem = [s for s in prem if s["prev"] in TRIGGER_PREV and s["next"] != "edit_file"]
        states = feb + prem if not SMOKE else feb[:4] + prem[:2]
        arms = dict(rep["arms"])
        srv = start_server(arms, os.path.join(ex.dir, "vllm_eval.log"))
        ex.log("vLLM ready with", sorted(arms))

        def holds_ref(new, st):
            rf = REF.get(st["id"]) or {}
            if not (rf.get("plus") or rf.get("minus")):
                return None
            have = {l.strip() for l in new.splitlines()}
            before = {l.strip() for l in st["content"].splitlines()}
            return all(p in have for p in rf["plus"]) and all(m not in have for m in rf["minus"] if m in before and m not in rf["plus"])

        def score(text, st):
            thought = text.split("<channel|>")[0]
            d = dict(quoted="```" in thought, opens_code=thought.lstrip().startswith("```") or thought.lstrip()[:1] in ("d", "c", "i", "f", "@"), ref=None,
                     n_thought=len(thought))
            m = CALL.search(text or "")
            if not m:
                return dict(d, o="no_call")
            if m.group(1) != "edit_file":
                return dict(d, o="other_tool")
            if st["kind"] == "premature":
                return dict(d, o="edit")
            a = dict(ARG.findall(m.group(2)))
            fp = (a.get("filepath") or "").removeprefix("/workspace/").lstrip("./")
            if fp != (st["filepath"] or "").removeprefix("/workspace/").lstrip("./"):
                return dict(d, o="other_file")
            old, new = a.get("old_string") or "", a.get("new_string") or ""
            if not old:
                return dict(d, o="empty")
            r = OFF.apply_replacement(st["content"], old, new, allow_multiple=False)
            if not r.error_message:
                return dict(d, o="applies", ref=holds_ref(r.new_content, st))
            return dict(d, o="noop" if old == new else ("multiple" if r.occurrences > 1 else "not_found"))
        models = [("base", "base", ""), ("forced_quote", "base", SENT["q"])] + [(k, k, "") for k in arms]
        work = [(arm, mdl, pre, k, s) for arm, mdl, pre in models for k, s in enumerate(states)]

        def one(w):
            arm, mdl, pre, k, s = w
            try:
                return arm, s["id"], score(answer(mdl, s["prompt"], 1000 + k, prefix=pre), s)
            except Exception as e:  # noqa: BLE001
                return arm, s["id"], dict(o="error:" + repr(e)[:80], quoted=False, opens_code=False, ref=None, n_thought=0)
        t1 = time.time()
        with ThreadPoolExecutor(128) as pool:
            rows_ = list(pool.map(one, work))
        ex.log("answers", len(rows_), "in", round(time.time() - t1), "s")
        srv.terminate()
        res = {}
        for arm, sid, r in rows_:
            res.setdefault(arm, {})[sid] = r
        ex.save_json("rows.json", res)
        fids = [s["id"] for s in states if s["kind"] == "feb"]
        pids = [s["id"] for s in states if s["kind"] == "premature"]
        refd = [i for i in fids if (REF.get(i) or {}).get("plus") or (REF.get(i) or {}).get("minus")]
        f_app = lambda r: r["o"] == "applies"  # noqa: E731
        f_ref = lambda r: bool(r["ref"])  # noqa: E731
        summ = {}
        for arm, v in res.items():
            c = {}
            for i in fids:
                c[v[i]["o"].split(":")[0]] = c.get(v[i]["o"].split(":")[0], 0) + 1
            row = dict(feb_counts=c, applies=sum(f_app(v[i]) for i in fids), holds_reference=sum(f_ref(v[i]) for i in refd), reference_states=len(refd),
                       quoted=sum(v[i]["quoted"] for i in fids), premature_edits=sum(v[i]["o"] == "edit" for i in pids), premature_states=len(pids),
                       median_thought_chars=sorted(v[i]["n_thought"] for i in fids)[len(fids) // 2])
            for other in ("base", "inj_n_L10"):
                if arm != other and other in res:
                    for key, f, pool_ in (("applies", f_app, fids), ("holds_reference", f_ref, refd), ("premature_edit", lambda r: r["o"] == "edit", pids)):
                        x = sum(1 for i in pool_ if f(v[i]) and not f(res[other][i]))
                        y = sum(1 for i in pool_ if f(res[other][i]) and not f(v[i]))
                        row[f"{key}_vs_{other}"] = (x, y, round(mcnemar(x, y), 4))
            summ[arm] = row
            ex.log(arm, row)
        ex.result(fit={k: v for k, v in rep.items() if k != "arms"}, summary=summ,
                  headline="compiled self-instruction (state injection) on FEB (" + str(len(fids)) + " states): applies / holds reference / quoted; premature edits of " +
                  str(len(pids)) + ": " + ", ".join(f"{a} {summ[a]['applies']}/{summ[a]['holds_reference']}/{summ[a]['quoted']}; {summ[a]['premature_edits']}" for a in summ))

"""EXP-118: a self-instruction circuit, synthetic neurons that make E2B open its thought with a sentence committing to copy the exact
lines it will change, then a code block (EXP-116's quote prefix), written into the weights in closed form.

Why: on FEB's 461 real first-edit states the forced quote prefix raised first edits that apply from 153 to 221 (against a neutral
sentence 159; 96 against 34 discordant, p < 0.001), while the code fence alone did nothing (145), so the words carry the effect and a
one-token neuron cannot (D-133). A phrase of K tokens needs K decision neurons.
Method (closed form, the down_proj of layers 28 to 34, rank K each, exact; one layer's write saturates at about 9.5 logits
because of the post-feedforward norm, so each neuron's write is shared by the three MLPs and calibrated on the model itself):
  trigger: the thought's first position after a read_file or search_similar_code response (code in view); 390 + 59 of FEB's states;
  fit data: thought-start states of dev-half tasks (tools/analysis/build_thought_starts.py) with the phrase appended (teacher forcing);
  neuron k (k = 0..K-1) reads the down_proj input at the position that predicts phrase token k (the trigger for k = 0, phrase token
        k-1 after it otherwise): weighted ridge, 1 at those positions in trigger states, 0 at 64 random positions per prompt, at the
        other phrase positions and at thought starts after other tools; its write raises phrase token k over the five tokens the model
        prefers there by c_k = max(0, median margin) + 3 logits in total (sne.conditional.write_vector, c_k / 7 per layer); then, on 48
        trigger states, the whole circuit is applied, the realised margins measured, and each neuron that leaves token k below the top
        in more than 10% of states is strengthened (up to 6 rounds);
  control: the same construction for EXP-116's neutral sentence (26 tokens), which was inert as a forced prefix.
Evaluation (official W4A16 E2B on vLLM with both adapters, official sampling, seed 1000 + index; thinking closed at 2,048 tokens then
512 more): FEB's 461 states (applies, holds the reference change, thought opens with the phrase), and after-read thought starts of
test-half tasks where E2B's recorded next call was not an edit (premature edits: edit calls made, and how many apply).
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
OUT = os.path.join(A, "EXP-118")
FIT_OUT = "/content/EXP-118_fit.json"
LAYER = 34
LAYERS = (28, 29, 30, 31, 32, 33, 34)  # one MLP's write saturates at about 9.5 logits (post-feedforward norm), so each write is shared
CALIB_STATES = 48
CALIB_ITERS = 6
MAX_FIT_TOKENS = 8192  # fit prompts keep their last 8,192 tokens (they end at the thought start), to bound activation memory
RANK = 32
NEG_PER_PROMPT = 64
PORT = 8023
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRIGGER_PREV = ("read_file", "search_similar_code")
PHRASES = {
    "quote": "Before editing, I will copy the exact lines I am going to change from the file content above, character for character:\n```python\n",
    "neutral": "Before editing, I will restate in my own words what the issue asks for and which behaviour of the code is wrong.\n",
}
OPENER = "<|channel>thought\n"
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)
LAST_TOOL = re.compile(r"<\|tool_response>response:(\w+)")


def last_tool(prompt):
    m = LAST_TOOL.findall(prompt)
    return m[-1] if m else None


def save_adapter(path, UV):
    """UV: {layer: (U [hidden, r] writes, V [r, d] read-outs)}, zero-padded to RANK; one PEFT adapter over the layers' down_proj."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    out = {}
    for L, (U, V) in UV.items():
        r = U.shape[1]
        Up = torch.zeros(U.shape[0], RANK)
        Vp = torch.zeros(RANK, V.shape[1])
        Up[:, :r], Vp[:r] = U, V
        base = f"base_model.model.model.language_model.layers.{L}.mlp.down_proj"
        out[f"{base}.lora_A.weight"] = Vp.float().contiguous()
        out[f"{base}.lora_B.weight"] = Up.float().contiguous()
    save_file(out, os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=RANK, lora_alpha=RANK,
                   target_modules=["down_proj"], layers_to_transform=sorted(UV), lora_dropout=0.0, bias="none", fan_in_fan_out=False,
                   init_lora_weights=True, inference_mode=True, use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


def ridge_primal(Xp, Xn, lam=1.0):
    """Class-balanced weighted ridge, v = argmin sum_i w_i (x_i.v - t_i)^2 + lam' |v|^2 with lam' = lam times the mean weighted squared
    norm of the rows (the same regulariser as sne.conditional.fit_readout's dual form); solved in the dual when there are fewer rows
    than dimensions, in the primal otherwise (identical solutions)."""
    X = torch.cat([Xp, Xn]).to(DEV).float()
    t = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).to(DEV)
    w = torch.ones(len(X), device=DEV)
    w[: len(Xp)] = len(Xn) / max(1, len(Xp))
    lam_abs = lam * float((w * X.pow(2).sum(1)).mean())
    if len(X) <= X.shape[1]:
        s = w.sqrt()
        Xs = (X * s[:, None]).double()
        alpha = torch.linalg.solve(Xs @ Xs.T + lam_abs * torch.eye(len(X), device=DEV, dtype=torch.float64), (s * t).double())
        v = (Xs.T @ alpha).float().cpu()
    else:
        G = (X * w[:, None]).T @ X
        G += lam_abs * torch.eye(G.shape[0], device=DEV)
        v = torch.linalg.solve(G, (X * w[:, None]).T @ t).float().cpu()  # float32: the ridge term keeps G well conditioned
        del G
    del X
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return v


def fit_stage():
    from collections import Counter
    from huggingface_hub import snapshot_download
    from sne.conditional import separation, write_vector
    from sne.model import layers, load
    rows = [json.loads(l) for l in open(os.path.join(A, "EXP-117", "thought_starts_dev.jsonl"), encoding="utf-8")]
    if SMOKE:
        rows = rows[:12]
    model, tok = load(os.environ.get("EXP118_MODEL") or snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat"),
                      ple_on_cpu=False, dev=DEV, **({"dtype": torch.float32} if os.environ.get("EXP118_FP32") else {}))
    opener = tok(OPENER, add_special_tokens=False)["input_ids"]
    Ls = layers(model)
    st = {}
    hs = []
    for L in LAYERS:
        hs += [Ls[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: st.__setitem__(("a", L), a[0][0].half())),
               Ls[L].mlp.register_forward_hook(lambda m, a, o, L=L: st.__setitem__(("y", L), o[0].float())),
               Ls[L].register_forward_hook(lambda m, a, o, L=L: st.__setitem__(("h", L), (o[0] if isinstance(o, tuple) else o)[0].float()))]
    rng = random.Random(118)
    rep = dict(layers=list(LAYERS))
    trig = [r for r in rows if r["prev"] in TRIGGER_PREV]
    other = [r for r in rows if r["prev"] not in TRIGGER_PREV]
    calib = trig[:CALIB_STATES]
    rep.update(trigger_states=len(trig), other_states=len(other), calibration_states=len(calib))

    def ids_of(r, P):
        ids = [tok.bos_token_id] + tok(r["prompt"], add_special_tokens=False)["input_ids"][-(MAX_FIT_TOKENS - 1):]
        assert ids[-3:] == opener
        return ids, len(ids) - 1, ids + P

    def margins_with(P, deltas):
        """Median over calibration states of (phrase token logit minus the best other) per position, and hit counts, with deltas added."""
        W0 = {L: Ls[L].mlp.down_proj.weight.data.clone() for L in deltas}
        for L, D in deltas.items():
            Ls[L].mlp.down_proj.weight.data = (W0[L].float() + D.to(W0[L].device)).to(W0[L].dtype)
        K = len(P)
        per = [[] for _ in range(K)]
        with torch.no_grad():
            for r in calib:
                ids, t0, full = ids_of(r, P)
                lg = model(input_ids=torch.tensor([full], device=DEV), use_cache=False, logits_to_keep=K + 1).logits[0].float()
                st.clear()
                for k in range(K):
                    row = lg[k].clone()
                    t = float(row[P[k]])
                    row[P[k]] = -1e9
                    per[k].append(t - float(row.max()))
                del lg
        for L in deltas:
            Ls[L].mlp.down_proj.weight.data = W0[L]
        med_ = [sorted(x)[len(x) // 2] for x in per]
        hit = [sum(v > 0 for v in x) for x in per]
        return med_, hit

    for name, phrase in PHRASES.items():
        P = tok(phrase, add_special_tokens=False)["input_ids"]
        K = len(P)
        pos = {L: [[] for _ in range(K)] for L in LAYERS}
        stepneg = {L: [[] for _ in range(K)] for L in LAYERS}
        neg = {L: [] for L in LAYERS}
        rms_y = {L: [[] for _ in range(K)] for L in LAYERS}
        rms_h = {L: [[] for _ in range(K)] for L in LAYERS}
        top5, margins = [[] for _ in range(K)], [[] for _ in range(K)]
        with torch.no_grad():
            for k_, r in enumerate(trig + other):
                is_trig = r["prev"] in TRIGGER_PREV
                ids, t0, full = ids_of(r, P)
                full = full if is_trig else ids
                out = model(input_ids=torch.tensor([full], device=DEV), use_cache=False, logits_to_keep=(K + 1) if is_trig else 1)
                starts = {i for i in range(2, len(ids)) if ids[i - 2:i + 1] == opener}
                pool = [i for i in range(1, t0) if i not in starts]
                pick = torch.tensor(rng.sample(pool, min(NEG_PER_PROMPT, len(pool))))
                for L in LAYERS:
                    act = st[("a", L)]
                    neg[L].append(act[pick.to(act.device)].half().cpu())
                    if is_trig:
                        for k in range(K):
                            p = t0 + k
                            a_p = act[p].half().cpu()  # one tensor, shared by reference across the lists below
                            pos[L][k].append(a_p)
                            for j in range(K):
                                if j != k:
                                    stepneg[L][j].append(a_p)
                            rms_y[L][k].append(float(st[("y", L)][p].pow(2).mean().sqrt()))
                            rms_h[L][k].append(float(st[("h", L)][p].pow(2).mean().sqrt()))
                    else:
                        a0 = act[t0].half().cpu()  # a thought start after another tool: negative for every neuron
                        for j in range(K):
                            stepneg[L][j].append(a0)
                if is_trig:
                    lg = out.logits[0].float()
                    for k in range(K):
                        row = lg[k].clone()
                        tgt = float(row[P[k]])
                        row[P[k]] = -1e9
                        top = torch.topk(row, 5).indices.tolist()
                        top5[k].extend(top)
                        margins[k].append(float(row[top[0]]) - tgt)
                del out
                st.clear()
                if k_ % 40 == 0:
                    if DEV == "cuda":
                        torch.cuda.empty_cache()
                    print(f"[gather] {name} {k_}/{len(trig) + len(other)}", flush=True)
        V = {L: [] for L in LAYERS}
        info = []
        b_ids = [[t for t, _ in Counter(top5[k]).most_common(5)] for k in range(K)]
        gap = [sorted(margins[k])[len(margins[k]) // 2] for k in range(K)]
        for L in LAYERS:
            Xneg = torch.cat(neg[L]).float()
            for k in range(K):
                Xp = torch.stack(pos[L][k]).float()
                Xn = torch.cat([Xneg, torch.stack(stepneg[L][k]).float()])
                v = ridge_primal(Xp, Xn)
                V[L].append(v)
                if L == LAYERS[-1]:
                    sep = separation(v, Xp, Xn)
                    info.append(dict(k=k, token=tok.decode([P[k]]), gap=round(gap[k], 3), pos_mean=sep["pos_mean"], pos_min=sep["pos_min"],
                                     neg_p99=sep["neg_p99"], neg_p999=sep["neg_p999"], neg_max=sep["neg_max"]))
            del Xneg
        med = lambda x: sorted(x)[len(x) // 2]  # noqa: E731
        c = [max(0.0, g) + 3.0 for g in gap]  # total logits wanted per position, shared equally by the layers

        def build(c):
            UV = {}
            for L in LAYERS:
                U = torch.stack([write_vector(model, L, P[k], b_ids[k], c[k] / len(LAYERS), med(rms_y[L][k]), med(rms_h[L][k])) for k in range(K)], 1)
                UV[L] = (U, torch.stack(V[L]))
            return UV
        calib_log = []
        for it in range(CALIB_ITERS):
            UV = build(c)
            m, hit = margins_with(P, {L: (U @ Vv) for L, (U, Vv) in UV.items()})
            calib_log.append(dict(iteration=it, c=[round(x, 2) for x in c], median_margin=[round(x, 2) for x in m], hits=hit, of=len(calib)))
            print(f"[calib] {name} it={it} hits {hit} of {len(calib)} | median margins {[round(x, 1) for x in m]}", flush=True)
            if all(h >= 0.9 * len(calib) for h in hit) or it == CALIB_ITERS - 1:
                break
            for k in range(K):
                if hit[k] < 0.9 * len(calib):
                    gain = m[k] + gap[k]  # realised change of the margin
                    c[k] *= min(3.0, max(1.25, (gap[k] + 2.0) / max(gain, 0.5)))
        path = save_adapter(os.path.join(OUT, f"circuit_{name}"), UV)
        for d, k in zip(info, range(K)):
            d.update(c_total=round(c[k], 3), final_median_margin=round(m[k], 3), final_hits=hit[k])
        rep[name] = dict(tokens=K, path=path, neurons=info, calibration=calib_log)
    for x in hs:
        x.remove()
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


if __name__ == "__main__" and os.environ.get("EXP118_STAGE") == "fit":
    fit_stage()
elif __name__ == "__main__":
    with Experiment("EXP-118", "self-instruction-circuit", category="improvement", depends_on=["EXP-116", "EXP-080"],
                    motivation="A forced thought prefix committing E2B to copy the exact lines raised first edits that apply by 44% on real "
                               "states; the code fence alone was inert, so the circuit must write the whole sentence.",
                    hypothesis="The quote circuit reproduces most of the forced prefix's gain in first edits that apply on FEB, above unedited "
                               "and above the neutral-sentence circuit, without making first edits less correct.",
                    config=dict(layers=list(LAYERS), calibration_states=CALIB_STATES, calibration_iterations=CALIB_ITERS, rank=RANK, neg_per_prompt=NEG_PER_PROMPT, trigger_prev=list(TRIGGER_PREV), phrases=PHRASES)) as ex:
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        flog = "/content/EXP-118_fit.log"
        with open(flog, "w") as fl:
            fit = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, EXP118_STAGE="fit"), stdout=fl, stderr=subprocess.STDOUT)
        ex.log("fit stage exit", fit.returncode, "in", round(time.time() - t0), "s; log tail:", open(flog).read()[-6000:])
        rep = json.load(open(FIT_OUT))
        ex.save_json("fit.json", rep)
        OFF = official()
        REF = json.load(open(os.path.join(A, "FEB", "feb_ref.json"), encoding="utf-8"))
        feb = [dict(json.loads(l), half=h, kind="feb") for h in ("dev", "test") for l in open(os.path.join(A, "FEB", f"feb_{h}.jsonl"), encoding="utf-8")]
        for s in feb:
            if s["prompt"].endswith("<|turn>model\n"):
                s["prompt"] += OPENER
        prem = [dict(json.loads(l), kind="premature") for l in open(os.path.join(A, "EXP-117", "thought_starts_test.jsonl"), encoding="utf-8")]
        prem = [s for s in prem if s["prev"] in TRIGGER_PREV and s["next"] != "edit_file"]
        states = feb + prem
        if SMOKE:
            states = feb[:4] + prem[:2]
        ex.log("states: FEB", sum(s["kind"] == "feb" for s in states), "| after-read, not edit-bound", sum(s["kind"] == "premature" for s in states))
        arms = {"circuit_quote": rep["quote"]["path"], "circuit_neutral": rep["neutral"]["path"]}
        md = next(p for p in __import__("glob").glob("/content/kaggle_input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/*") if os.path.exists(os.path.join(p, "config.json")))
        cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--host", "127.0.0.1",
               "--port", str(PORT), "--max-model-len", "32768", "--gpu-memory-utilization", "0.88", "--enforce-eager", "--max-num-seqs", "128",
               "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0}), "--enable-lora", "--max-lora-rank", str(RANK), "--max-loras", str(len(arms)),
               "--max-cpu-loras", str(len(arms)), "--lora-modules", *[f"{k}={v}" for k, v in arms.items()]]
        log = os.path.join(ex.dir, "vllm.log")
        srv = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        while True:
            if srv.poll() is not None:
                raise RuntimeError("vLLM exited: " + open(log, errors="replace").read()[-4000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(5)
        ex.log("vLLM ready")

        def req(body):
            r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=3600) as f:
                return json.loads(f.read())

        def answer(model, prompt, seed):
            r = req(dict(model=model, prompt=prompt, max_tokens=2048, temperature=0.2, top_p=0.95, seed=seed, skip_special_tokens=False, stop=["<tool_call|>"]))
            t = r["choices"][0]["text"]
            if r["choices"][0].get("finish_reason") == "length" and "<channel|>" not in t and "<|tool_call>" not in t:
                r2 = req(dict(model=model, prompt=prompt + t + "<channel|>", max_tokens=512, temperature=0.2, top_p=0.95, seed=seed, skip_special_tokens=False,
                              stop=["<tool_call|>"]))
                t = t + "<channel|>" + r2["choices"][0]["text"]
            return t

        def holds_ref(new, st):
            rf = REF.get(st["id"]) or {}
            if not (rf.get("plus") or rf.get("minus")):
                return None
            have = {l.strip() for l in new.splitlines()}
            before = {l.strip() for l in st["content"].splitlines()}
            return all(p in have for p in rf["plus"]) and all(m not in have for m in rf["minus"] if m in before and m not in rf["plus"])

        def score(text, st):
            thought = text.split("<channel|>")[0]
            d = dict(quoted="```" in thought, opens_quote=thought.startswith(PHRASES["quote"][:40]), opens_neutral=thought.startswith(PHRASES["neutral"][:40]), ref=None)
            m = CALL.search(text or "")
            if not m:
                return dict(d, o="no_call")
            if m.group(1) != "edit_file":
                return dict(d, o="other_tool")
            if st["kind"] == "premature":
                return dict(d, o="edit")  # an edit call where E2B's recorded next call was not an edit; the file is not known here
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
        models = [("base", "base")] + [(k, k) for k in arms]
        work = [(arm, mdl, k, s) for arm, mdl in models for k, s in enumerate(states)]

        def one(w):
            arm, mdl, k, s = w
            try:
                return arm, s["id"], score(answer(mdl, s["prompt"], 1000 + k), s)
            except Exception as e:  # noqa: BLE001
                return arm, s["id"], dict(o="error:" + repr(e)[:80], quoted=False, opens_quote=False, opens_neutral=False, ref=None)
        t1 = time.time()
        with ThreadPoolExecutor(128) as pool:
            rows = list(pool.map(one, work))
        ex.log("answers", len(rows), "in", round(time.time() - t1), "s")
        srv.terminate()
        res = {}
        for arm, sid, r in rows:
            res.setdefault(arm, {})[sid] = r
        ex.save_json("rows.json", res)
        fids = [s["id"] for s in states if s["kind"] == "feb"]
        pids = [s["id"] for s in states if s["kind"] == "premature"]
        trig_ids = [s["id"] for s in states if s["kind"] == "feb" and last_tool(s["prompt"]) in TRIGGER_PREV]
        refd = [i for i in fids if (REF.get(i) or {}).get("plus") or (REF.get(i) or {}).get("minus")]
        f_app = lambda r: r["o"] == "applies"  # noqa: E731
        f_ref = lambda r: bool(r["ref"])  # noqa: E731
        summ = {}
        for arm, v in res.items():
            c = {}
            for i in fids:
                c[v[i]["o"].split(":")[0]] = c.get(v[i]["o"].split(":")[0], 0) + 1
            row = dict(feb_counts=c, applies=sum(f_app(v[i]) for i in fids), holds_reference=sum(f_ref(v[i]) for i in refd), reference_states=len(refd),
                       opens_quote=sum(v[i]["opens_quote"] for i in fids), opens_quote_trigger_states=sum(v[i]["opens_quote"] for i in trig_ids),
                       opens_neutral=sum(v[i]["opens_neutral"] for i in fids), trigger_states=len(trig_ids),
                       premature_edits=sum(v[i]["o"] == "edit" for i in pids), premature_states=len(pids),
                       premature_opens_phrase=sum(v[i]["opens_quote"] or v[i]["opens_neutral"] for i in pids))
            for other in ("base", "circuit_neutral"):
                if arm != other and other in res:
                    for key, f, pool_ in (("applies", f_app, fids), ("holds_reference", f_ref, refd), ("premature_edit", lambda r: r["o"] == "edit", pids)):
                        x = sum(1 for i in pool_ if f(v[i]) and not f(res[other][i]))
                        y = sum(1 for i in pool_ if f(res[other][i]) and not f(v[i]))
                        row[f"{key}_vs_{other}"] = (x, y, round(mcnemar(x, y), 4))
            summ[arm] = row
            ex.log(arm, row)
        ex.result(fit={k: {kk: vv for kk, vv in v.items() if kk != "neurons"} if isinstance(v, dict) else v for k, v in rep.items()}, summary=summ,
                  headline="self-instruction circuit on FEB (" + str(len(fids)) + " states): applies / holds reference / opens with the phrase; "
                  "premature edits of " + str(len(pids)) + ": " + ", ".join(f"{a} {summ[a]['applies']}/{summ[a]['holds_reference']}/{summ[a]['opens_quote'] + summ[a]['opens_neutral']}; {summ[a]['premature_edits']}" for a in summ))

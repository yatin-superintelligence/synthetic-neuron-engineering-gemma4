"""EXP-092: patch-correctness neurons for the E2B agent, training-free, measured on a verified patch bench.

The agent's dominant failure is a wrong patch (EXP-044a: 36 of 58 tasks submit a patch that fails tests). This experiment aims a
synthetic neuron at that step. It has three parts.
1. The patch bench. The verified injected-bug tasks (tools/trainset/kaggle_build.py; none inside a function named in the 58
   evaluation fixes) become agent states at the moment of the fix, in the official format: the official system prompt and
   swegemma's own task prompt and workspace tree, then the failing tests run (the real failing output), a grep for the function
   (real), and read_file of the buggy code (the real file). Each state has one correct edit, known exactly. An answer is scored
   two ways. It applies if its old_string occurs exactly once in the code shown. It is correct if applying it gives the same code
   as applying the reference fix, with trailing spaces ignored.
2. Directions. E2B samples 4 answers per fitting state (official W4A16, temperature 1.0). In states with both outcomes, E2B's
   activations on the tool-call tokens of its correct answers are contrasted with those of its incorrect ones (bf16, teacher
   forced). This is the agent's own correct-minus-incorrect direction at the edit.
3. A neuron at layer L: down_proj += u v^T (an exact rank-1 LoRA). Read-out v is about 1 on tool-call tokens and about 0 on
   thought and prompt tokens. Write u shifts the residual after layer L by alpha * |delta mu| along the direction, calibrated by a
   forward pass, as in EXP-089.

Pre-registered (D-062, before any result):
  Splits: tasks split 50/25/25 into fit, dev and test (seed 92).
  Bench answers: greedy, at most 2,048 new tokens. If the thought is still open at the cap, it is closed and the answer continues
    for at most 1,024 more tokens (as the official thinking budget does).
  Layers: L in {9, 12, 15, 18, 21, 24, 27}. Two are kept: those with the best held-out pairwise accuracy (fit states with both
    outcomes, split 2:1 within the fit set).
  Dev scan: the unedited model, and alpha in {1, 3, 9} at each kept layer. Selection: most correct (ties: smaller alpha, earlier
    layer).
  Test: the unedited model, the selected neuron, and a control neuron (same read-out, write along a random orthogonal direction
    calibrated to the same shift).
  Gate for official agent runs: on test, the selected neuron's correct count beats the unedited model's (exact McNemar,
    two-sided, p < 0.05) and beats the control neuron's.
The bench's baseline (E2B's greedy correct and apply rates on all states) is itself a result; a 0 or 100% baseline would make
the bench useless and is reported as such.
"""
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

_src = open(os.path.join(os.path.dirname(__file__), "exp088_sft_lora.py"), encoding="utf-8").read()
G = {"__file__": os.path.join(os.path.dirname(__file__), "exp088_sft_lora.py"), "__name__": "exp088"}
exec(compile(_src.split('with Experiment("EXP-088"')[0], "exp088", "exec"), G)  # materialize, trajectory, render (tested in EXP-088)

LAYERS = [9, 12, 15, 18, 21, 24, 27]
ALPHAS = [1, 3, 9]
OUT = os.path.join(ROOT, "artifacts_local", "EXP-092")
PORT = 8013
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)


def official_model():
    return next(p for p in glob.glob("/content/kaggle_input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/*") if os.path.exists(os.path.join(p, "config.json")))


def load_tasks():
    files = sorted(glob.glob(os.path.join(ROOT, "artifacts_local", "trainset", "*", "trainset", "tasks.jsonl")))
    uniq = {}
    for f in files:
        for l in open(f, encoding="utf-8"):
            t = json.loads(l)
            if t.get("source") != "synthetic" or not isinstance(t.get("buggy_excerpt"), dict) or "content" not in t["buggy_excerpt"]:
                continue
            key = hashlib.sha1(t["bug_patch"].encode("utf-8")).hexdigest()[:8]
            if key not in uniq:
                t["instance_id"] = f"{t['instance_id']}_{key}"
                uniq[key] = t
    return list(uniq.values()), files


def build_states(tasks, ex):
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.harness.agent_runner import build_workspace_tree_command
    COMP = G["COMP"]
    cfg = EvalConfig(tasks_path=os.path.join(COMP, "tasks.jsonl"), snapshots_dir=os.path.join(COMP, "snapshots"), results_dir="/content/sft/v",
                     submission_dir="/content/sft", models={}, sandbox="subprocess", timeout_seconds=300, verification_timeout_seconds=300,
                     max_time_minutes=60, max_tool_calls=100, max_turns=None, limits=build_submission_limits()[0],
                     generation_constraints=build_submission_limits()[1], verbose=False, concurrency=1)
    sysp = next((p for p in glob.glob(os.path.join(COMP, "**", "prompts", "system.md"), recursive=True)), None)
    system = open(sysp, encoding="utf-8").read()
    os.makedirs(G["WORK"], exist_ok=True)
    by = {}
    for t in tasks:
        by.setdefault((t["repo"], t["base_instance"]), []).append(t)
    states, skipped = [], {}
    for (repo, base), ts in by.items():
        d = G["materialize"](base, repo)
        tree = G["run"](build_workspace_tree_command(workspace_dir=d), d).stdout
        for t in ts:
            bp = os.path.join(G["WORK"], "bug.patch")
            open(bp, "w", encoding="utf-8").write(t["bug_patch"])
            if G["run"](f"git apply {bp}", d).returncode != 0:
                skipped["bug patch does not apply"] = skipped.get("bug patch does not apply", 0) + 1
                continue
            try:
                m = G["trajectory"](t, d, tree, system, cfg)
            finally:
                G["run"](f"git apply -R {bp}", d)
            if m is None:
                skipped["no single-line fix"] = skipped.get("no single-line fix", 0) + 1
                continue
            k = next(i for i, x in enumerate(m) if x["role"] == "assistant" and x["tool_calls"][0]["function"]["name"] == "edit_file")
            gold = m[k]["tool_calls"][0]["function"]["arguments"]
            exc = t["buggy_excerpt"]["content"]
            if exc.count(gold["old_string"]) != 1:
                skipped["reference edit not unique in the excerpt"] = skipped.get("reference edit not unique in the excerpt", 0) + 1
                continue
            states.append(dict(iid=t["instance_id"], repo=repo, operator=t["operator"], prefix=m[:k], gold=gold, excerpt=exc))
        ex.log(repo, "states so far", len(states), "skipped", skipped)
    return states, skipped


def norm(s):
    return "\n".join(l.rstrip() for l in s.split("\n"))


def score(text, st):
    m = CALL.search(text or "")
    if not m:
        return dict(tool=None, applies=False, correct=False)
    args = dict(ARG.findall(m.group(2)))
    if m.group(1) != "edit_file":
        return dict(tool=m.group(1), applies=False, correct=False)
    old, new = args.get("old_string"), args.get("new_string")
    if not old or new is None or st["excerpt"].count(old) != 1:
        return dict(tool="edit_file", applies=False, correct=False)
    got = st["excerpt"].replace(old, new, 1)
    want = st["excerpt"].replace(st["gold"]["old_string"], st["gold"]["new_string"], 1)
    return dict(tool="edit_file", applies=True, correct=norm(got) == norm(want))


class Server:
    """The official vLLM on the official W4A16 checkpoint, with our rank-1 LoRAs served as extra model names."""

    def __init__(self, loras, log):
        md = official_model()
        cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--host", "127.0.0.1",
               "--port", str(PORT), "--max-model-len", "32768", "--gpu-memory-utilization", "0.80", "--enforce-eager", "--max-num-seqs", "128",
               "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]
        if loras:
            cmd += ["--enable-lora", "--max-lora-rank", "8", "--max-loras", str(len(loras)), "--max-cpu-loras", str(len(loras)), "--lora-modules",
                    *[f"{n}={p}" for n, p in loras.items()]]
        self.p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 1200:
            if self.p.poll() is not None:
                raise RuntimeError("vLLM exited: " + open(log, errors="replace").read()[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                return
            except Exception:  # noqa: BLE001
                time.sleep(5)
        raise RuntimeError("vLLM not ready in 20 min")

    def _req(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=3600))

    def answers(self, model, prompts, temperature=0.0, n=1, seed=0):
        """Each prompt's n answers; a thought still open at 2,048 tokens is closed and the answer continues for 1,024 tokens."""
        def one(p):
            r = self._req(dict(model=model, prompt=p, max_tokens=2048, temperature=temperature, n=n, seed=seed, skip_special_tokens=False))
            outs = []
            for c in r["choices"]:
                t = c["text"]
                if c.get("finish_reason") == "length" and "<channel|>" not in t and "<|tool_call>" not in t:
                    r2 = self._req(dict(model=model, prompt=p + t + "<channel|>", max_tokens=1024, temperature=temperature, n=1, seed=seed,
                                        skip_special_tokens=False))
                    t = t + "<channel|>" + r2["choices"][0]["text"]
                outs.append(t)
            return outs
        def safe(p):  # one rejected request (for example an over-long prompt) is scored as no answer, not a crash of the whole bench
            try:
                return one(p)
            except Exception as e:  # noqa: BLE001
                self.errors.append(repr(e)[:200])
                return [""] * n
        self.errors = getattr(self, "errors", [])
        with ThreadPoolExecutor(128) as ex:  # vLLM runs up to 128 at once; at 64 the KV cache was 25% used (EXP-092 scan)
            return list(ex.map(safe, prompts))

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(60)
        except subprocess.TimeoutExpired:
            self.p.kill()
        time.sleep(5)


def save_neuron(path, L, u, v):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    base = f"base_model.model.model.language_model.layers.{L}.mlp.down_proj"
    save_file({f"{base}.lora_A.weight": v.view(1, -1).float().contiguous(), f"{base}.lora_B.weight": u.view(-1, 1).float().contiguous()},
              os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=1, lora_alpha=1, target_modules=["down_proj"],
                   layers_to_transform=[L], lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
                   use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


def directions(tok, fit_states, samples, ex):
    """Teacher-forced bf16 activations over the tool-call tokens of E2B's own answers; paired correct-minus-incorrect per layer."""
    from huggingface_hub import snapshot_download
    from sne.conditional import fit_readout, separation
    from sne.model import layers, load
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False)
    Ls = layers(model)
    st = {}
    hooks = [Ls[L].register_forward_hook(lambda m, a, o, L=L: st.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float())) for L in LAYERS]
    rows = []
    with torch.no_grad():
        for k, (s, ans) in enumerate(samples):
            prompt = G["render"](tok, s["prefix"], True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(ans["text"], add_special_tokens=False)["input_ids"]
            text_call = ans["text"].find("<|tool_call>")
            if text_call < 0:
                continue
            c0 = len(p_ids) + len(tok(ans["text"][:text_call], add_special_tokens=False)["input_ids"])
            ids = torch.tensor([p_ids + a_ids], device="cuda")
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
            rows.append(dict(state=s["iid"], ok=ans["correct"], c0=c0, n_prompt=len(p_ids), R={L: st[L][c0:].mean(0).cpu() for L in LAYERS}))
            if k % 100 == 0:
                print(f"[patch] activations {k}/{len(samples)}", flush=True)
    for h in hooks:
        h.remove()
    by = {}
    for r in rows:
        by.setdefault(r["state"], {True: [], False: []})[r["ok"]].append(r)
    mixed = sorted(sid for sid, v in by.items() if v[True] and v[False])
    if len(mixed) < 6:
        raise RuntimeError(f"only {len(mixed)} fit states have both a correct and an incorrect sample; no direction can be fitted")
    rng = random.Random(92)
    rng.shuffle(mixed)
    held = set(mixed[: len(mixed) // 3])
    rep = dict(samples=len(rows), mixed_states=len(mixed), held_mixed=len(held), layers={})
    dirs = {}
    for L in LAYERS:
        diffs = [torch.stack([r["R"][L] for r in by[sid][True]]).mean(0) - torch.stack([r["R"][L] for r in by[sid][False]]).mean(0)
                 for sid in mixed if sid not in held]
        d = torch.stack(diffs).mean(0)
        dn = d / d.norm()
        wins = sum(float(torch.stack([r["R"][L] @ dn for r in by[sid][True]]).mean() > torch.stack([r["R"][L] @ dn for r in by[sid][False]]).mean())
                   for sid in held)
        rep["layers"][L] = dict(pairwise_acc=round(wins / max(1, len(held)), 4), dmu=round(float(d.norm()), 4))
        dirs[L] = (dn, float(d.norm()))
        print(f"[patch] layer {L} {rep['layers'][L]}", flush=True)
    sel = sorted(LAYERS, key=lambda L: -rep["layers"][L]["pairwise_acc"])[:2]
    rep["selected_layers"] = sel
    # read-out (tool-call tokens 1, thought and prompt tokens 0), MLP output scale, calibrated writes
    acts = {L: dict(pos=[], neg=[], rms=[]) for L in sel}
    hk = []
    for L in sel:
        hk.append(Ls[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: st.__setitem__(("a", L), a[0][0].float())))
        hk.append(Ls[L].mlp.register_forward_hook(lambda m, a, o, L=L: st.__setitem__(("y", L), o[0].float())))
    g = torch.Generator().manual_seed(92)
    pick = rng.sample(samples, min(160, len(samples)))
    fitted = []
    with torch.no_grad():
        for s, ans in pick:
            tc = ans["text"].find("<|tool_call>")
            if tc < 0:
                continue
            prompt = G["render"](tok, s["prefix"], True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c0 = len(p_ids) + len(tok(ans["text"][:tc], add_special_tokens=False)["input_ids"])
            ids = torch.tensor([p_ids + tok(ans["text"], add_special_tokens=False)["input_ids"]], device="cuda")
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
            fitted.append((s, ans, c0))
            for L in sel:
                a = st[("a", L)].cpu()
                pos, neg = a[c0:], a[1:c0]
                acts[L]["pos"].append(pos[torch.randperm(len(pos), generator=g)[:40]].half())
                acts[L]["neg"].append(neg[torch.randperm(len(neg), generator=g)[:40]].half())
                acts[L]["rms"].append(float(st[("y", L)][c0:].pow(2).mean(-1).sqrt().median()))
    for h in hk:
        h.remove()
    vec = {}
    for L in sel:
        n = len(acts[L]["pos"])
        cut = int(n * 0.8)
        v = fit_readout(torch.cat(acts[L]["pos"][:cut]).float(), torch.cat(acts[L]["neg"][:cut]).float(), lam=1.0)
        sep = separation(v, torch.cat(acts[L]["pos"][cut:]).float(), torch.cat(acts[L]["neg"][cut:]).float())
        dn, dmu = dirs[L]
        w = Ls[L].post_feedforward_layernorm.weight.float().cpu()
        sc = float(Ls[L].layer_scalar.float().cpu())
        rms_y = float(torch.tensor(acts[L]["rms"]).median())
        r = torch.randn(dn.shape, generator=g)
        r = r - (r @ dn) * dn
        rn = r / r.norm()

        def measured(u, direction):
            vv, uu = v.cuda(), u.cuda()
            res = {}
            h1 = Ls[L].register_forward_hook(lambda m, a, o: res.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
            sh = []
            with torch.no_grad():
                for s, ans, c0 in fitted[-10:]:
                    ids = torch.tensor([tok(G["render"](tok, s["prefix"], True), add_special_tokens=False)["input_ids"]
                                        + tok(ans["text"], add_special_tokens=False)["input_ids"]], device="cuda")
                    model(input_ids=ids, use_cache=False, logits_to_keep=1)
                    h0 = res["h"][c0:]
                    h2 = Ls[L].mlp.down_proj.register_forward_hook(lambda m, a, o: o + (a[0].float() @ vv)[..., None].to(o.dtype) * uu.to(o.dtype))
                    model(input_ids=ids, use_cache=False, logits_to_keep=1)
                    h2.remove()
                    sh.append(float(((res["h"][c0:] - h0) @ direction).mean()))
            h1.remove()
            return sum(sh) / len(sh)
        u0 = dmu * dn * rms_y / (sc * w)
        m0 = measured(u0, dn)
        u1 = u0 * (dmu / m0) if abs(m0) > 1e-9 else u0
        uc0 = dmu * rn * rms_y / (sc * w)
        mc = measured(uc0, rn)
        uc1 = uc0 * (dmu / mc) if abs(mc) > 1e-9 else uc0
        vec[L] = dict(v=v, u1=u1, uc1=uc1)
        rep["layers"][L].update(readout_check=sep, rms_y=round(rms_y, 4), first_order_shift=round(m0, 4), target=round(dmu, 4), control_first_order=round(mc, 4))
        print(f"[patch] layer {L} neuron {rep['layers'][L]}", flush=True)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return vec, rep


def mcnemar(a, b):
    import math
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    n = x + y
    return x, y, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n))


with Experiment("EXP-092", "patch-correctness-neurons", category="improvement", depends_on=["EXP-089", "EXP-088"],
                motivation="The agent mostly fails by writing wrong patches; a verified bench of fix states measures that step directly, and a neuron "
                           "built from E2B's own correct-minus-incorrect edits targets it without training.",
                hypothesis="A patch-correctness neuron raises E2B's correct-edit rate on held-out bench states against the unedited model (exact "
                           "McNemar p < 0.05) and against a control neuron of matched strength.",
                config=dict(layers=LAYERS, alphas=ALPHAS, splits="fit 50 / dev 25 / test 25, seed 92", samples_per_fit_state=4, fit_temperature=1.0,
                            answer_tokens="2,048 then thought closed and 1,024 more", model="official E2B W4A16 on vLLM 0.19.1")) as ex:
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tasks, files = load_tasks()
    ex.log("verified tasks", len(tasks), "from", files)
    if SMOKE:
        tasks = tasks[:24]
    states, skipped = build_states(tasks, ex)
    tok = AutoTokenizer.from_pretrained(official_model())
    rng = random.Random(92)
    ids = sorted({s["iid"] for s in states})
    rng.shuffle(ids)
    n = len(ids)
    split = {i: ("fit" if k < n // 2 else "dev" if k < 3 * n // 4 else "test") for k, i in enumerate(ids)}
    for s in states:
        s["split"] = split[s["iid"]]
    json.dump([{k: v for k, v in s.items() if k != "prefix"} for s in states], open(os.path.join(OUT, "states.json"), "w"))
    def vprompt(prefix):  # vLLM adds the BOS token itself, so the rendered "<bos>" is dropped (the HF pass keeps it)
        p = G["render"](tok, prefix, True)
        return p[len("<bos>"):] if p.startswith("<bos>") else p
    prompts = {s["iid"]: vprompt(s["prefix"]) for s in states}
    ex.log("states", len(states), "skipped", skipped, {sp: sum(s["split"] == sp for s in states) for sp in ("fit", "dev", "test")},
           "| median prompt tokens", sorted(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts.values())[len(prompts) // 2])
    # 1. baseline on all states (greedy) and 4 samples per fit state; a rerun reuses both when saved (the first run of 2026-11-08 saved
    # them, then stopped before the scan because the HF model still held GPU memory when vLLM started)
    fit = [s for s in states if s["split"] == "fit"]
    bpath, spath = os.path.join(OUT, "base_greedy.json"), os.path.join(OUT, "fit_samples.json")
    resumed = os.path.exists(bpath) and os.path.exists(spath)
    if resumed:
        base = json.load(open(bpath, encoding="utf-8"))
        by_iid = {s["iid"]: s for s in fit}
        saved = [a for a in json.load(open(spath, encoding="utf-8")) if a["iid"] in by_iid]
        ex.log("resumed: baseline and fit samples loaded from the earlier run", len(base), len(saved))
    else:
        srv = Server({}, os.path.join(OUT, "vllm_base.log"))
        try:
            greedy = srv.answers("base", [prompts[s["iid"]] for s in states])
            sampled = srv.answers("base", [prompts[s["iid"]] for s in fit], temperature=1.0, n=4, seed=92)
        finally:
            srv.close()
        ex.log("requests rejected by vLLM (scored as no answer):", len(getattr(srv, "errors", [])), getattr(srv, "errors", [])[:3])
        base = {s["iid"]: dict(score(g[0], s), text=g[0][-1500:]) for s, g in zip(states, greedy)}
        json.dump(base, open(bpath, "w"))
    by_split = {sp: [base[s["iid"]] for s in states if s["split"] == sp] for sp in ("fit", "dev", "test")}
    tools = {}
    for b in base.values():
        tools[b["tool"]] = tools.get(b["tool"], 0) + 1
    ex.log("baseline greedy: correct", {sp: f"{sum(x['correct'] for x in v)}/{len(v)}" for sp, v in by_split.items()}, "| applies",
           {sp: f"{sum(x['applies'] for x in v)}/{len(v)}" for sp, v in by_split.items()}, "| next tool", tools)
    samples = []
    if resumed:
        samples = [(by_iid[a["iid"]], dict(text=a["text"], correct=a["correct"], applies=a["applies"])) for a in saved]
    else:
        for s, outs in zip(fit, sampled):
            for t in outs:
                sc_ = score(t, s)
                samples.append((s, dict(text=t, correct=sc_["correct"], applies=sc_["applies"])))
        json.dump([dict(iid=s["iid"], **a) for s, a in samples], open(spath, "w"))  # reused by EXP-093
    ex.log("fit samples", len(samples), "correct", sum(a["correct"] for _, a in samples), "applies", sum(a["applies"] for _, a in samples))
    # 2. directions and neurons
    vec, rep = directions(tok, fit, samples, ex)
    import gc
    gc.collect()
    torch.cuda.empty_cache()  # the HF model must leave the GPU before vLLM starts (2026-11-08: only 16.3 of 22 GB free otherwise)
    torch.save(vec, os.path.join(OUT, "vectors.pt"))
    ex.save_json("directions.json", rep)
    sel = rep["selected_layers"]
    loras = {}
    for L in sel:
        for a in ALPHAS:
            loras[f"n_L{L}_a{a}"] = save_neuron(os.path.join(OUT, f"n_L{L}_a{a}"), L, a * vec[L]["u1"], vec[L]["v"])
    dev = [s for s in states if s["split"] == "dev"]
    srv = Server(loras, os.path.join(OUT, "vllm_scan.log"))
    try:
        scan = {name: [score(o[0], s)["correct"] for s, o in zip(dev, srv.answers(name, [prompts[s["iid"]] for s in dev]))] for name in loras}
    finally:
        srv.close()
    scan["base"] = [base[s["iid"]]["correct"] for s in dev]
    counts = {k: sum(v) for k, v in scan.items()}
    ex.log("dev scan (correct of", len(dev), "):", counts)
    best = max(loras, key=lambda nme: (counts[nme], -int(nme.split("_a")[1]), -sel.index(int(nme.split("_L")[1].split("_")[0]))))
    L, a = int(best.split("_L")[1].split("_")[0]), int(best.split("_a")[1])
    ctrl = save_neuron(os.path.join(OUT, "control"), L, a * vec[L]["uc1"], vec[L]["v"])
    test = [s for s in states if s["split"] == "test"]
    srv = Server({"selected": loras[best], "control": ctrl}, os.path.join(OUT, "vllm_test.log"))
    try:
        res = {nme: [score(o[0], s) for s, o in zip(test, srv.answers(nme, [prompts[s["iid"]] for s in test]))] for nme in ("selected", "control")}
    finally:
        srv.close()
    tb = [base[s["iid"]]["correct"] for s in test]
    tsel = [r["correct"] for r in res["selected"]]
    tctl = [r["correct"] for r in res["control"]]
    x, y, p = mcnemar(tsel, tb)
    cx, cy, cp = mcnemar(tsel, tctl)
    gate = p < 0.05 and x > y and sum(tsel) > sum(tctl)
    ex.result(states=len(states), skipped=skipped, baseline={sp: dict(correct=sum(x_["correct"] for x_ in v), applies=sum(x_["applies"] for x_ in v), n=len(v))
                                                              for sp, v in by_split.items()},
              next_tool=tools, directions=rep, dev_scan=counts, selected=best,
              test=dict(n=len(test), base=sum(tb), selected=sum(tsel), control=sum(tctl), selected_only=x, base_only=y, p=round(p, 4),
                        vs_control=dict(selected_only=cx, control_only=cy, p=round(cp, 4)),
                        applies=dict(selected=sum(r["applies"] for r in res["selected"]), control=sum(r["applies"] for r in res["control"]))),
              gate_passed=gate,
              headline=f"patch bench: unedited correct {sum(tb)}/{len(test)} on test; {best} {sum(tsel)} (control {sum(tctl)}), McNemar p = {round(p, 4)}; gate "
                       f"{'passed' if gate else 'not passed'}")

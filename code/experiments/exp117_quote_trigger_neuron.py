"""EXP-117: a closed-form synthetic neuron that makes E2B open its thought with a code block (a quote of the code), tested on FEB.

Why: E2B's thought quotes code before 77% of first edits that apply and 44% of those that fail (E4B about 90% of both); EXP-116
tests whether forcing the quote causes edits to apply. A single decision neuron can supply the quote's first token, '```' (one token,
id 2717), at the thought's first position, which is a fixed token pattern ('<|channel>', 'thought', newline).
Method (EXP-076/080/113's closed-form decision neuron at layer 34, rank 1 on down_proj; a layer-34 write changes only the logits of
the position where it fires):
  data: thought-start states from E2B's official runs, dev-half tasks only (tools/analysis/build_thought_starts.py): 348 prompts
        ending at the opened thought, 116 each after a read_file response, after another tool and at a new model turn;
  read-out (weighted ridge in the primal, on the down_proj input): fence_all, 1 at every thought start, 0 at 64 random other
        positions per prompt (thought starts in the history are left out of both); fence_read, 1 at thought starts after a read_file
        response, 0 at the other thought starts and at the random positions;
  write: raises '```' over the mean of the five commonest first thought tokens by c logits where the read-out is 1
        (sne.conditional.write_vector); c = the median margin of the top token over '```' at the positive states, plus 3;
  control: the fence_all read-out with the same c raising 'Before' (a sentence opening, as EXP-116's neutral arm) instead of '```'.
Evaluation: all 461 FEB states (official W4A16 E2B on vLLM with the adapters, official sampling, seed 1000 + index), arms base,
fence_all, fence_read, control, zero LoRA. Measures as EXP-116: first edits that apply (official matcher), edits after which the file
holds the reference change, thoughts that quote code, thoughts whose first token is '```'. Separation of each read-out on the
test-half thought starts (unseen tasks) is reported.
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
OUT = os.path.join(A, "EXP-117")
FIT_OUT = "/content/EXP-117_fit.json"
LAYER = 34
NEG_PER_PROMPT = 64
PORT = 8022
FENCE = 2717
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)


def save_adapter(path, u, v):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    base = f"base_model.model.model.language_model.layers.{LAYER}.mlp.down_proj"
    save_file({f"{base}.lora_A.weight": v[None, :].float().contiguous(), f"{base}.lora_B.weight": u[:, None].float().contiguous()},
              os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=1, lora_alpha=1, target_modules=["down_proj"],
                   layers_to_transform=[LAYER], lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
                   use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


def ridge_primal(Xp, Xn, lam=1.0):
    """Weighted ridge with balanced classes, solved in the primal on the GPU: v = (X^T W X + lam' I)^-1 X^T W t."""
    X = torch.cat([Xp, Xn]).cuda().float()
    t = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).cuda()
    w = torch.ones(len(X), device="cuda")
    w[: len(Xp)] = len(Xn) / max(1, len(Xp))
    G = (X * w[:, None]).T @ X
    G += lam * G.diagonal().mean() * torch.eye(G.shape[0], device="cuda")
    return torch.linalg.solve(G.double(), ((X * w[:, None]).T @ t).double()).float().cpu()


def fit_stage():
    from huggingface_hub import snapshot_download
    from sne.conditional import separation, write_vector
    from sne.model import layers, load
    rows = {h: [json.loads(l) for l in open(os.path.join(A, "EXP-117", f"thought_starts_{h}.jsonl"), encoding="utf-8")] for h in ("dev", "test")}
    if SMOKE:
        rows = {h: v[:9] for h, v in rows.items()}
    model, tok = load(snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat"), ple_on_cpu=False)
    opener = tok("<|channel>thought\n", add_special_tokens=False)["input_ids"]
    before_id = tok("Before", add_special_tokens=False)["input_ids"][0]
    Ls = layers(model)
    st = {}
    hs = [Ls[LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0].float())),
          Ls[LAYER].mlp.register_forward_hook(lambda m, a, o: st.__setitem__("y", o[0, -1].float())),
          Ls[LAYER].register_forward_hook(lambda m, a, o: st.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))]
    rng = random.Random(117)
    first_tok = {}
    with torch.no_grad():
        for h, rs in rows.items():
            for k, r in enumerate(rs):
                ids = [tok.bos_token_id] + tok(r["prompt"], add_special_tokens=False)["input_ids"]
                assert ids[-3:] == opener, "prompt must end at the opened thought"
                starts = {i for i in range(2, len(ids)) if ids[i - 2:i + 1] == opener}  # every thought start, history included
                out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
                lg = out.logits[0, -1].float()
                top = int(lg.argmax())
                first_tok[top] = first_tok.get(top, 0) + 1
                pool = [i for i in range(1, len(ids) - 1) if i not in starts]
                neg_idx = rng.sample(pool, min(NEG_PER_PROMPT, len(pool)))
                r.update(a=st["a"][-1].half().cpu(), neg=st["a"][torch.tensor(neg_idx, device=st["a"].device)].half().cpu(),
                         rms_y=float(st["y"].pow(2).mean().sqrt()), rms_h=float(st["h"].pow(2).mean().sqrt()),
                         margin=float(lg[top] - lg[FENCE]), margin_before=float(lg[top] - lg[before_id]), top=top)
                del out
                if k % 40 == 0:
                    torch.cuda.empty_cache()
                    print(f"[gather] {h} {k}/{len(rs)}", flush=True)
    for x in hs:
        x.remove()
    dev, test = rows["dev"], rows["test"]
    common = [t for t, _ in sorted(first_tok.items(), key=lambda x: -x[1]) if t != FENCE][:5]
    rep = dict(dev=len(dev), test=len(test), common_first_tokens=[(tok.decode([t]), first_tok[t]) for t in common],
               fence_first_top=first_tok.get(FENCE, 0))
    Xneg = torch.cat([r["neg"].float() for r in dev])
    Xall = torch.stack([r["a"].float() for r in dev])
    v_all = ridge_primal(Xall, Xneg)
    Xr = torch.stack([r["a"].float() for r in dev if r["prev_class"] == "read_file"])
    Xo = torch.cat([torch.stack([r["a"].float() for r in dev if r["prev_class"] != "read_file"]), Xneg])
    v_read = ridge_primal(Xr, Xo)
    tneg = torch.cat([r["neg"].float() for r in test])
    for name, vv, pos_fn in (("fence_all", v_all, lambda r: True), ("fence_read", v_read, lambda r: r["prev_class"] == "read_file")):
        tp = torch.stack([r["a"].float() for r in test if pos_fn(r)])
        tn = torch.cat([torch.stack([r["a"].float() for r in test if not pos_fn(r)]), tneg]) if not all(pos_fn(r) for r in test) else tneg
        rep[f"{name}_separation_test"] = separation(vv, tp, tn)
    m = sorted(r["margin"] for r in dev)
    c = m[len(m) // 2] + 3.0
    rms_y = sorted(r["rms_y"] for r in dev)[len(dev) // 2]
    rms_h = sorted(r["rms_h"] for r in dev)[len(dev) // 2]
    rep.update(c=round(c, 3), median_margin_top_over_fence=round(m[len(m) // 2], 3),
               median_margin_top_over_before=round(sorted(r["margin_before"] for r in dev)[len(dev) // 2], 3))
    arms = {}
    for name, vv, a_id in (("fence_all", v_all, FENCE), ("fence_read", v_read, FENCE), ("control", v_all, before_id)):
        u = write_vector(model, LAYER, a_id, common, c, rms_y, rms_h)
        arms[name] = save_adapter(os.path.join(OUT, name), u, vv)
    rep["arms"] = arms
    print("[fit]", json.dumps({k: v_ for k, v_ in rep.items() if k != "arms"}), flush=True)
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


if __name__ == "__main__" and os.environ.get("EXP117_STAGE") == "fit":
    fit_stage()
elif __name__ == "__main__":
    with Experiment("EXP-117", "quote-trigger-neuron", category="improvement", depends_on=["EXP-116", "EXP-080", "EXP-113"],
                    motivation="E2B's first edits apply far more often when its thought quotes the code first; one synthetic neuron can open "
                               "the thought with a code block.",
                    hypothesis="On real first-edit states the fence neuron raises thoughts that quote code and first edits that apply, above "
                               "unedited and above the same read-out writing a sentence opening.",
                    config=dict(layer=LAYER, neg_per_prompt=NEG_PER_PROMPT, arms=["fence_all", "fence_read", "control", "zero"])) as ex:
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        flog = "/content/EXP-117_fit.log"
        with open(flog, "w") as fl:
            fit = subprocess.run([sys.executable, os.path.abspath(__file__)] + (["--smoke"] if SMOKE else []), env=dict(os.environ, EXP117_STAGE="fit"),
                                 stdout=fl, stderr=subprocess.STDOUT)
        ex.log("fit stage exit", fit.returncode, "in", round(time.time() - t0), "s; log tail:", open(flog).read()[-3000:])
        rep = json.load(open(FIT_OUT))
        ex.save_json("fit.json", rep)
        OFF = official()
        REF = json.load(open(os.path.join(A, "FEB", "feb_ref.json"), encoding="utf-8"))
        SOLVABLE = {"fastapi_14786", "rich_2943", "requests_6644", "fastapi_14301", "rich_3006", "rich_3454", "fastapi_9425"}
        states = [dict(json.loads(l), half=h) for h in ("dev", "test") for l in open(os.path.join(A, "FEB", f"feb_{h}.jsonl"), encoding="utf-8")]
        for s in states:  # 39 of 461 sit at a new model turn, where E2B writes the thought opener itself (as in EXP-116)
            if s["prompt"].endswith("<|turn>model\n"):
                s["prompt"] += "<|channel>thought\n"
        if SMOKE:
            states = states[:6]
        arms = dict(rep["arms"], zero=os.path.join(A, "fastlab_zero", "zero_v"))
        md = next(p for p in __import__("glob").glob("/content/kaggle_input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/*") if os.path.exists(os.path.join(p, "config.json")))
        cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--host", "127.0.0.1",
               "--port", str(PORT), "--max-model-len", "32768", "--gpu-memory-utilization", "0.88", "--enforce-eager", "--max-num-seqs", "128",
               "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0}), "--enable-lora", "--max-lora-rank", "8", "--max-loras", str(len(arms)),
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
            rf = REF[st["id"]]
            if not (rf["plus"] or rf["minus"]):
                return None
            have = {l.strip() for l in new.splitlines()}
            before = {l.strip() for l in st["content"].splitlines()}
            return all(p in have for p in rf["plus"]) and all(m not in have for m in rf["minus"] if m in before and m not in rf["plus"])

        def score(text, st):
            thought = text.split("<channel|>")[0]
            d = dict(quoted="```" in thought, fence_first=thought.startswith("```"), ref=None)
            m = CALL.search(text or "")
            if not m:
                return dict(d, o="no_call")
            if m.group(1) != "edit_file":
                return dict(d, o="read_file" if m.group(1) == "read_file" else "other_tool")
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
                return arm, s["id"], dict(o="error:" + repr(e)[:80], quoted=False, fence_first=False, ref=None)
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
        refd = [i for i in ids if REF[i]["plus"] or REF[i]["minus"]]
        solv = [s["id"] for s in states if s["iid"] in SOLVABLE]
        summ = {}
        f_app = lambda r: r["o"] == "applies"  # noqa: E731
        f_ref = lambda r: bool(r["ref"])  # noqa: E731
        for arm, v in res.items():
            c = {}
            for r in v.values():
                c[r["o"].split(":")[0]] = c.get(r["o"].split(":")[0], 0) + 1
            row = dict(counts=c, applies=sum(f_app(v[i]) for i in ids), holds_reference=sum(f_ref(v[i]) for i in refd), reference_states=len(refd),
                       quoted=sum(v[i]["quoted"] for i in ids), fence_first=sum(v[i]["fence_first"] for i in ids),
                       applies_solvable=sum(f_app(v[i]) for i in solv), solvable_states=len(solv))
            for other in ("base", "control"):
                if arm != other and other in res:
                    for key, f, pool_ in (("applies", f_app, ids), ("holds_reference", f_ref, refd)):
                        x = sum(1 for i in pool_ if f(v[i]) and not f(res[other][i]))
                        y = sum(1 for i in pool_ if f(res[other][i]) and not f(v[i]))
                        row[f"{key}_vs_{other}"] = (x, y, round(mcnemar(x, y), 4))
            summ[arm] = row
            ex.log(arm, row)
        ex.result(fit={k: v_ for k, v_ in rep.items() if k != "arms"}, summary=summ, headline="quote-trigger neuron on FEB (" + str(len(states)) +
                  " states), applies / holds reference / thought opens with a fence: " +
                  ", ".join(f"{a} {summ[a]['applies']}/{summ[a]['holds_reference']}/{summ[a]['fence_first']}" for a in summ))

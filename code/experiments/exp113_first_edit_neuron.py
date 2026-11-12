"""EXP-113: a closed-form neuron that detects a first edit about to fail and makes E2B read instead.

Why: on solvable tasks a first edit that applies leads to a solve in 72% of runs, one that fails in 14% (tools/analysis/first_edit.py);
on real first-edit states (FEB) 47% of recorded first edits fail, mostly because E2B edits code it has not read (placeholders, assumed
lines, '# ... existing code ...', no-op edits; tools/analysis/feb_failures.py). The decision is made in the reasoning before the tool
name, so the state at the tool-name token should show whether the coming edit is grounded.
Method (EXP-076/080's closed-form decision neuron, at layer 34):
  data: each FEB state plus E2B's recorded reasoning, ending at '<|tool_call>call:' (tools/analysis/build_feb_decisions.py), labelled
        by whether that recorded edit applied (official matcher on the file as it stood): dev half 230 (121 applied), test half 231;
  read-out: weighted ridge on the down_proj input at that token, 1 where the edit failed, 0 where it applied, fitted on dev only;
        its separation on the test half's unseen tasks is reported (AUC);
  write: raises read_file's first tool-name token over edit_file's by c logits where the read-out is 1 (sne.conditional.write_vector),
        c = the median edit-over-read margin at dev failing states plus 1;
  arms (official W4A16 E2B on vLLM, FEB test states, official sampling): neuron at c and at c/2, the same write on a read-out fitted to
        shuffled labels (control), a zero LoRA, unedited. Outcomes per state: applies, failed edit kinds, read_file, other.
Pass for an official run: fewer failing first edits than unedited and control (McNemar p < 0.05) with applying edits not lower.
"""
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import importlib.util
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

A = os.path.join(ROOT, "artifacts_local")
OUT = os.path.join(A, "EXP-113")
FIT_OUT = "/content/EXP-113_fit.json"
LAYER = 34
PORT = 8021
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)


def first_token(tok, name):
    a = tok("<|tool_call>call:", add_special_tokens=False)["input_ids"]
    b = tok("<|tool_call>call:" + name, add_special_tokens=False)["input_ids"]
    assert b[:len(a)] == a, name
    return b[len(a)]


def auc(pos, neg):
    pos, neg = sorted(pos), sorted(neg)
    if not pos or not neg:
        return None
    wins = 0.0
    j = 0
    for p in pos:  # fraction of (pos, neg) pairs ordered correctly
        while j < len(neg) and neg[j] < p:
            j += 1
        k = j
        ties = 0
        while k < len(neg) and neg[k] == p:
            ties += 1
            k += 1
        wins += j + 0.5 * ties
    return round(wins / (len(pos) * len(neg)), 4)


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


def fit_stage():
    from huggingface_hub import snapshot_download
    from sne.conditional import fit_readout, write_vector
    from sne.model import layers, load
    rows = {h: [json.loads(l) for l in open(os.path.join(A, "FEB", f"feb_decision_{h}.jsonl"), encoding="utf-8")] for h in ("dev", "test")}
    if SMOKE:
        rows = {h: v[:12] for h, v in rows.items()}
    model, tok = load(snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat"), ple_on_cpu=False)
    rid, eid = first_token(tok, "read_file"), first_token(tok, "edit_file")
    Ls = layers(model)
    st = {}
    hs = [Ls[LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0, -1].float().cpu())),
          Ls[LAYER].mlp.register_forward_hook(lambda m, a, o: st.__setitem__("y", o[0, -1].float())),
          Ls[LAYER].register_forward_hook(lambda m, a, o: st.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))]
    with torch.no_grad():
        for h, rs in rows.items():
            for k, r in enumerate(rs):
                ids = tok(r["decision"], add_special_tokens=False)["input_ids"]
                out = model(input_ids=torch.tensor([[tok.bos_token_id] + ids], device="cuda"), use_cache=False, logits_to_keep=1)
                r.update(a=st["a"].half(), rms_y=float(st["y"].pow(2).mean().sqrt()), rms_h=float(st["h"].pow(2).mean().sqrt()),
                         margin=float(out.logits[0, -1, eid] - out.logits[0, -1, rid]))
                del out
                if k % 50 == 0:
                    torch.cuda.empty_cache()
                    print(f"[gather] {h} {k}/{len(rs)}", flush=True)
    for x in hs:
        x.remove()
    dev, test = rows["dev"], rows["test"]
    Xp = torch.stack([r["a"].float() for r in dev if r["label"] != "applies"])
    Xn = torch.stack([r["a"].float() for r in dev if r["label"] == "applies"])
    v = fit_readout(Xp, Xn, lam=1.0)
    # control: the same fit on shuffled labels (same counts)
    g = random.Random(113)
    lab = [r["label"] != "applies" for r in dev]
    g.shuffle(lab)
    Xa = torch.stack([r["a"].float() for r in dev])
    vc = fit_readout(Xa[torch.tensor(lab)], Xa[~torch.tensor(lab)], lam=1.0)
    score = lambda vv, rs: [float(r["a"].float() @ vv) for r in rs]  # noqa: E731
    rep = dict(dev=len(dev), test=len(test), dev_failed=int(Xp.shape[0]), test_failed=sum(r["label"] != "applies" for r in test))
    for name, vv in (("neuron", v), ("control", vc)):
        st_ = score(vv, test)
        rep[f"{name}_auc_test"] = auc([s for s, r in zip(st_, test) if r["label"] != "applies"], [s for s, r in zip(st_, test) if r["label"] == "applies"])
        sd = score(vv, dev)
        rep[f"{name}_auc_dev"] = auc([s for s, r in zip(sd, dev) if r["label"] != "applies"], [s for s, r in zip(sd, dev) if r["label"] == "applies"])
        rep[f"{name}_test_mean_failed"] = round(sum(s for s, r in zip(st_, test) if r["label"] != "applies") / max(1, rep["test_failed"]), 3)
        rep[f"{name}_test_mean_applied"] = round(sum(s for s, r in zip(st_, test) if r["label"] == "applies") / max(1, len(test) - rep["test_failed"]), 3)
    fm = sorted(r["margin"] for r in dev if r["label"] != "applies")
    c = (fm[len(fm) // 2] if fm else 5.0) + 1.0
    rms_y = sorted(r["rms_y"] for r in dev)[len(dev) // 2]
    rms_h = sorted(r["rms_h"] for r in dev)[len(dev) // 2]
    rep.update(c=round(c, 3), median_edit_over_read_margin_failed=round(fm[len(fm) // 2], 3) if fm else None)
    arms = {}
    for name, vv, cc in (("neuron", v, c), ("neuron_half", v, c / 2), ("control", vc, c)):
        u = write_vector(model, LAYER, rid, eid, cc, rms_y, rms_h)
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


if __name__ == "__main__" and os.environ.get("EXP113_STAGE") == "fit":
    fit_stage()
elif __name__ == "__main__":
    with Experiment("EXP-113", "first-edit-failure-neuron", category="improvement", depends_on=["EXP-112", "EXP-080"],
                    motivation="First edits that fail mostly come from editing unread code; a neuron that detects a failing first edit at the "
                               "tool-name token and writes read_file over edit_file would turn those into reads.",
                    hypothesis="On held-out tasks' first-edit states the neuron lowers failing first edits against unedited and its shuffled-label "
                               "control without lowering applying edits.",
                    config=dict(layer=LAYER, ridge=1.0, arms=["neuron", "neuron_half", "control", "zero"])) as ex:
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        flog = "/content/EXP-113_fit.log"
        with open(flog, "w") as fl:
            fit = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, EXP113_STAGE="fit"), stdout=fl, stderr=subprocess.STDOUT)
        ex.log("fit stage exit", fit.returncode, "in", round(time.time() - t0), "s; log tail:", open(flog).read()[-3000:])
        rep = json.load(open(FIT_OUT))
        ex.save_json("fit.json", rep)
        OFF = official()
        states = [json.loads(l) for l in open(os.path.join(A, "FEB", "feb_test.jsonl"), encoding="utf-8")]
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

        def score(text, st):
            m = CALL.search(text or "")
            if not m:
                return "no_call"
            if m.group(1) != "edit_file":
                return "read_file" if m.group(1) == "read_file" else "other_tool"
            a = dict(ARG.findall(m.group(2)))
            fp = (a.get("filepath") or "").removeprefix("/workspace/").lstrip("./")
            if fp != (st["filepath"] or "").removeprefix("/workspace/").lstrip("./"):
                return "edit_other_file"
            old, new = a.get("old_string") or "", a.get("new_string") or ""
            if not old:
                return "edit_failed"
            r = OFF.apply_replacement(st["content"], old, new, allow_multiple=False)
            return "applies" if not r.error_message else "edit_failed"
        models = [("base", "base")] + [(k, k) for k in arms]
        work = [(arm, mdl, k, s) for arm, mdl in models for k, s in enumerate(states)]

        def one(w):
            arm, mdl, k, s = w
            try:
                return arm, s["id"], score(answer(mdl, s["prompt"], 1000 + k), s)
            except Exception as e:  # noqa: BLE001
                return arm, s["id"], "error"
        with ThreadPoolExecutor(128) as pool:
            rows = list(pool.map(one, work))
        srv.terminate()
        res = {}
        for arm, sid, o in rows:
            res.setdefault(arm, {})[sid] = o
        ex.save_json("rows.json", res)
        ids = [s["id"] for s in states]
        summ = {}
        for arm, v in res.items():
            c = {}
            for o in v.values():
                c[o] = c.get(o, 0) + 1
            row = dict(counts=c)
            if arm != "base":
                for key in ("applies", "edit_failed"):
                    x = sum(1 for i in ids if v[i] == key and res["base"][i] != key)
                    y = sum(1 for i in ids if res["base"][i] == key and v[i] != key)
                    row[f"{key}_vs_base"] = (x, y, round(mcnemar(x, y), 4))
            summ[arm] = row
            ex.log(arm, row)
        ex.result(fit=rep, summary=summ, headline="first-edit failure neuron on held-out FEB states: " +
                  ", ".join(f"{a} applies {summ[a]['counts'].get('applies', 0)} failed {summ[a]['counts'].get('edit_failed', 0)} read {summ[a]['counts'].get('read_file', 0)}" for a in summ))

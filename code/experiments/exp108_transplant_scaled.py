"""EXP-108: the E4B transplant scaled from 7 to 18 layers, judged on reference-fix tokens of unseen tasks.

EXP-106 showed the first specific transfer: 896 E4B neurons (7 layers x 128, state map, strength 1) raised E2B's rank-1 accuracy on
reference-edit tokens of 248 held-out states from 89.10% to 90.09%, while random E4B neurons transplanted the same way gave 89.17%
(D-107). If the gain adds across layers, more layers should give more. This run repeats EXP-102's method (same fit states, maps,
selection and placement; imported from exp102_neuron_transplant.py) at 18 E2B layers (every second layer from 2 to 34, each paired
with the E4B layer at the same relative depth), state map only, and evaluates with EXP-106's measure:
  arms: st18_a1 (strength 1), st18_a2 (strength 2), st18_rnd_a1 (random E4B neurons, same filter, layers and strength);
  measure: rank-1 accuracy and log-probability of the reference token at every reference-edit token of 160 held-out states (dev and
  test of EXP-092's split; the transplant's fit states are never used), plus the unedited model, all in bf16 with each adapter
  merged exactly into E2B's weights.
Pass for an official run: held accuracy above EXP-106's 7-layer transplant (90.09%) and above the random control by more than
EXP-106's margin.
"""
import json
import math
import os
import random
import re
import runpy
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
X = runpy.run_path(os.path.join(HERE, "exp102_neuron_transplant.py"), run_name="x102lib")  # functions and helpers, no main block
# runpy returns a COPY of the module's globals; the functions read their own globals, so the overrides go there (2026-11-09: without
# this the first run silently refitted EXP-102's 7 layers)
G = X["fit_stage"].__globals__
G.update(L2=list(range(2, 35, 2)), LATE=[], ARMS=[("st18_a1", "st", 1.0), ("st18_a2", "st", 2.0), ("st18_rnd_a1", "st_rnd", 1.0)],
         EXP="EXP-108", FIT_OUT="/content/EXP-108_fit.json", LAB_DIR="/content/x108_adapters")
X = G
torch = X["torch"]
P = X["P"]
ROOT = X["ROOT"]
SMOKE = X["SMOKE"]
OUT = os.path.join(ROOT, "artifacts_local", "EXP-108")
N_EVAL = 24 if SMOKE else 160
STR = re.compile(r'(?:old_string|new_string):<\|"\|>(.*?)<\|"\|>', re.S)


def mcnemar(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def deltas(path):
    from safetensors.torch import load_file
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    out = {}
    for k, v in sd.items():
        if k.endswith(".lora_A.weight"):
            base = k[: -len(".lora_A.weight")]
            out[(int(base.split(".layers.")[1].split(".")[0]), base.rsplit(".", 1)[1])] = sd[base + ".lora_B.weight"].float() @ v.float()
    return out


if __name__ == "__main__" and os.environ.get("EXP108_STAGE") == "fit":
    X["fit_stage"]()
elif __name__ == "__main__":
    from sne.registry import Experiment
    with Experiment("EXP-108", "transplant-scaled-18-layers", category="improvement", depends_on=["EXP-102", "EXP-106"],
                    motivation="EXP-106's 7-layer E4B transplant moved reference-fix tokens of unseen tasks by a point beyond its control; "
                               "more layers may move them more.",
                    hypothesis="An 18-layer transplant raises held reference-edit token accuracy above the 7-layer one (90.09%) and above its random control.",
                    config=dict(e2b_layers=X["L2"], arms=[a[0] for a in X["ARMS"]], eval_states=N_EVAL, neurons_per_layer=X["K"])) as ex:
        import shutil
        from transformers import AutoTokenizer
        from sne.model import layers, load
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        flog = "/content/EXP-108_fit.log"
        with open(flog, "w") as fl:
            fit = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, EXP108_STAGE="fit"), stdout=fl, stderr=subprocess.STDOUT)
        ex.log("fit stage exit", fit.returncode, "in", round(time.time() - t0), "s; log tail:", open(flog).read()[-4000:])
        F = json.load(open(X["FIT_OUT"]))
        ex.save_json("transplant.json", F["rep"])
        arms = F["arms"]
        for name, rel in (("st7_a1", "artifacts_local/EXP-102/st_a1"), ("st7_rnd_a1", "artifacts_local/EXP-102/st_rnd_a1")):  # EXP-106's 7-layer pair, for
            if os.path.exists(os.path.join(ROOT, rel)):  # its significance test on the same held states (EXP-106 stopped before its tests)
                arms[name] = os.path.join(ROOT, rel)
        # evaluation states: held-out (dev and test), never the fit states
        log = ex.log
        tasks, _ = P["load_tasks"]()
        if SMOKE:
            tasks = tasks[:40]
        states, _ = P["build_states"](tasks, ex)
        tok = AutoTokenizer.from_pretrained(P["official_model"]())
        ids = sorted({s["iid"] for s in states})
        random.Random(92).shuffle(ids)
        held_ids = set(ids[len(ids) // 2:])
        held = [s for s in states if s["iid"] in held_ids]
        random.Random(108).shuffle(held)
        data = []
        for s in held:
            pre = P["G"]["render"](tok, s["prefix"], True)
            full = P["G"]["render"](tok, s["prefix"] + [X["gold_message"](s)], False)
            if not full.startswith(pre):
                continue
            ans = full[len(pre):]
            p_ids = tok(pre, add_special_tokens=False)["input_ids"]
            enc = tok(ans, add_special_tokens=False, return_offsets_mapping=True)
            a_ids, offs = enc["input_ids"], enc["offset_mapping"]
            spans = [m.span(1) for m in STR.finditer(ans)]
            tgt = [j for j, (a, b) in enumerate(offs) if b > a and any(x <= a and b <= y for x, y in spans)]
            if not tgt or len(p_ids) + len(a_ids) > 14000:
                continue
            data.append(dict(iid=s["iid"], ids=p_ids + a_ids, pos=[len(p_ids) + j - 1 for j in tgt], gold=[a_ids[j] for j in tgt]))
            if len(data) >= N_EVAL:
                break
        log("eval states", len(data), "| reference-edit tokens", sum(len(d["pos"]) for d in data))
        from huggingface_hub import snapshot_download
        model, _ = load(snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat"), ple_on_cpu=False)
        Ls = layers(model)
        tm = model.model.language_model if hasattr(model.model, "language_model") else model.model
        E = model.get_output_embeddings().weight
        cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
        st = {}
        hk = tm.norm.register_forward_hook(lambda m, a, o: st.__setitem__("hf", o[0]))

        def evaluate(tag):
            rows = []
            with torch.no_grad():
                for d in data:
                    model(input_ids=torch.tensor([d["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
                    z = st["hf"][torch.tensor(d["pos"], device="cuda")].float() @ E.float().T
                    if cap:
                        z = cap * torch.tanh(z / cap)
                    lp = torch.log_softmax(z, -1)
                    g = torch.tensor(d["gold"], device="cuda")
                    rows.append(dict(iid=d["iid"], right=(z.argmax(-1) == g).tolist(), lp=lp[torch.arange(len(g)), g].tolist()))
            acc = sum(sum(r["right"]) for r in rows) / max(1, sum(len(r["right"]) for r in rows))
            log(tag, "held accuracy", round(acc, 4))
            return rows
        res = {"base": evaluate("base")}
        for arm, path in arms.items():
            D = deltas(path)
            orig = {}
            for (L, mod), dW in D.items():
                w = getattr(Ls[L].mlp, mod).weight
                orig[(L, mod)] = w.data.clone()
                w.data = (w.data.float() + dW.to(w.device)).to(w.dtype)
            res[arm] = evaluate(arm)
            for (L, mod), w0 in orig.items():
                getattr(Ls[L].mlp, mod).weight.data = w0
            del orig
            if arm.startswith("st18"):
                shutil.copytree(path, os.path.join(OUT, arm), dirs_exist_ok=True)
        hk.remove()
        ex.save_json("rows.json", res)
        summ = {}
        B = res["base"]
        for arm, R in res.items():
            tot = sum(len(r["right"]) for r in R)
            row = dict(tokens=tot, accuracy=round(sum(sum(r["right"]) for r in R) / max(1, tot), 4), mean_logprob=round(sum(sum(r["lp"]) for r in R) / max(1, tot), 4))
            if arm != "base":
                x = sum(1 for r, b in zip(R, B) for i, j in zip(r["right"], b["right"]) if i and not j)
                y = sum(1 for r, b in zip(R, B) for i, j in zip(r["right"], b["right"]) if j and not i)
                row.update(tokens_gained=x, tokens_lost=y, p_tokens=round(mcnemar(x, y), 4))
            summ[arm] = row
            log(arm, row)
        ex.result(summary=summ, layers=F["rep"]["layers"], headline="18-layer E4B transplant, held reference-edit token accuracy: " +
                  ", ".join(f"{a} {summ[a]['accuracy']}" for a in summ))

"""EXP-104: an instruction compiled into neurons. The effect of the lab's best instruction ('all3', D-098) written into one MLP layer of
E2B in closed form, so the unedited prompt behaves as if the instruction were there (context distillation without training).

Why: lab v1 found that a three-part instruction (one small mistake; edit at once; copy old_string exactly, change only the wrong token)
raised E2B's applying edits from 27 to 40 of 96 states, the only result outside the measured noise floor (D-099). An instruction
changes the model's internal states at every later position; if a weight edit reproduces that change, the behaviour moves into the
weights - a neuron-engineering result in its own right, and a way to give the agent the behaviour without the prompt.
Method, at E2B layer L (10, 14 or 20):
  record: 160 fit states of EXP-092's bench with the reference fix written in, run twice in bf16: as they are, and with the
          instruction appended to the system message. Tokens after the system message are the same in both runs; at those aligned
          positions (the reference-answer tokens, the last 16 prompt tokens and 128 random ones per state) keep, from the plain run,
          the down_proj input k and output y at layer L, and from both runs the layer output h;
  target: the change d = h(instructed) - h(plain) at each aligned position, taken back through layer L's post-feedforward norm and
          layer scalar into a down_proj output change dy (64 positions per state from the system message's shared start, where d is
          exactly 0, keep the edit quiet where nothing should change);
  solve:  ridge least squares dW = argmin |K dW^T - dY|^2 + lam |dW|^2, then its best rank-128 approximation as an exact down_proj LoRA.
Arms (fast lab, D-082/D-099, quick 96 states, unedited arm and the instruction itself as a prompt arm in the same server):
  ins_L10, ins_L14, ins_L20 (strength 1), ins_L14_x2 (strength 2);
  controls: ctl_L10, ctl_L14, ctl_L20 - the same solve with a neutral instruction of about the same length (no fixing advice).
"""
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

HERE = os.path.dirname(__file__)
_src = open(os.path.join(HERE, "exp092_patch_neurons.py"), encoding="utf-8").read()
P = {"__file__": os.path.join(HERE, "exp092_patch_neurons.py"), "__name__": "exp092"}
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)
EXP = "EXP-104"
OUT = os.path.join(ROOT, "artifacts_local", EXP)
LAB_DIR = "/content/fastlab/adapters/x104"
FIT_OUT = f"/content/{EXP}_fit.json"
INSTR = ("The bug is one small mistake on one line. Once you have read the code that contains it, fix it right away with edit_file. "
         "Copy old_string character for character from the file content you read, without adding escape characters, and in new_string "
         "change only the wrong token or operator. Do not rewrite or restructure code.")
CTRL = ("The repository is a Python project. Its files use four spaces for indentation and its tests are written with pytest. "
        "The tools listed above are the only ones available in this session, and every tool call returns its result as JSON. "
        "Paths are given relative to the repository root, which is also the working directory of every command you run.")
LAY = [10, 14, 20]
RANK = 128
N_STATES = 24 if SMOKE else 160
N_RAND, N_LAST, N_ZERO = 128, 16, 64
ARMS = [("ins_L10", "ins", 10, 1.0), ("ins_L14", "ins", 14, 1.0), ("ins_L20", "ins", 20, 1.0), ("ins_L14_x2", "ins", 14, 2.0),
        ("ctl_L10", "ctl", 10, 1.0), ("ctl_L14", "ctl", 14, 1.0), ("ctl_L20", "ctl", 20, 1.0)]


def gold_message(st):
    return {"role": "assistant", "content": "", "reasoning": "This matches the failing test. I will make this one-line fix.",
            "tool_calls": [{"type": "function", "function": {"name": "edit_file", "arguments": st["gold"]}}]}


def with_system(prefix, txt):
    return [dict(m, content=m["content"] + chr(10) * 2 + txt) if m.get("role") == "system" else m for m in prefix]


def inv_scale(w):
    e = (0.05 * w.abs().median()) ** 2
    return w / (w * w + e)


def mcnemar(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def save_lora(path, L, A, B):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    base = f"base_model.model.model.language_model.layers.{L}.mlp.down_proj"
    save_file({f"{base}.lora_A.weight": A.to(torch.bfloat16).contiguous(), f"{base}.lora_B.weight": B.to(torch.bfloat16).contiguous()},
              os.path.join(path, "adapter_model.safetensors"))
    r = A.shape[0]
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=r, lora_alpha=r, target_modules=["down_proj"],
                   layers_to_transform=[L], lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
                   use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


def fit_stage():
    import gc
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from sne.model import layers, load
    log = lambda *a: print(time.strftime("[%H:%M:%S]"), *a, flush=True)  # noqa: E731
    ex = type("X", (), {"log": staticmethod(log)})()
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:40]
    states, _ = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    ids = sorted({s["iid"] for s in states})
    random.Random(92).shuffle(ids)
    fit = set(ids[: len(ids) // 2])
    fit_states = [s for s in states if s["iid"] in fit]
    random.Random(104).shuffle(fit_states)
    rng = random.Random(104)
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False)
    Ls = layers(model)
    st = {}
    hooks = []
    def keep_down(m, a, o, L):  # returns None: a forward hook that returns a value replaces the module's output (2026-11-09 failure)
        st[("k", L)] = a[0][0]
        st[("y", L)] = o[0]

    def keep_layer(m, a, o, L):
        st[("h", L)] = (o[0] if isinstance(o, tuple) else o)[0]
    for L in LAY:
        hooks.append(Ls[L].mlp.down_proj.register_forward_hook(lambda m, a, o, L=L: keep_down(m, a, o, L)))
        hooks.append(Ls[L].register_forward_hook(lambda m, a, o, L=L: keep_layer(m, a, o, L)))
    rec = {L: dict(K=[], Y=[], D={"ins": [], "ctl": []}) for L in LAY}
    n_used = 0
    with torch.no_grad():
        for s in fit_states:
            msgs = s["prefix"] + [gold_message(s)]
            seqs = {}
            for tag, txt in (("plain", None), ("ins", INSTR), ("ctl", CTRL)):
                m = msgs if txt is None else with_system(s["prefix"], txt) + [gold_message(s)]
                seqs[tag] = tok(P["G"]["render"](tok, m, False), add_special_tokens=False)["input_ids"]
            a = seqs["plain"]
            n_ans = len(tok(P["G"]["render"](tok, msgs, False), add_special_tokens=False)["input_ids"]) - len(
                tok(P["G"]["render"](tok, s["prefix"], True), add_special_tokens=False)["input_ids"])
            # shared start (system message before the appended text) and shared end (everything after the system message)
            ok = True
            pos = {}
            for tag in ("ins", "ctl"):
                b = seqs[tag]
                pre = 0
                while pre < min(len(a), len(b)) and a[pre] == b[pre]:
                    pre += 1
                suf = 0
                while suf < min(len(a), len(b)) - pre and a[-1 - suf] == b[-1 - suf]:
                    suf += 1
                if suf < n_ans + N_LAST + N_RAND or len(a) + 0 > 14000:
                    ok = False
                pos[tag] = (pre, suf, len(b) - len(a))
            if not ok:
                continue
            suf = min(pos["ins"][1], pos["ctl"][1])
            pre = min(pos["ins"][0], pos["ctl"][0])
            start = len(a) - suf
            tail = list(range(len(a) - n_ans - N_LAST, len(a) - 1))
            rnd = rng.sample(range(start, len(a) - n_ans - N_LAST), min(N_RAND, len(a) - n_ans - N_LAST - start))
            zero = rng.sample(range(1, pre), min(N_ZERO, pre - 1))
            P_al = torch.tensor(sorted(set(tail + rnd)), device="cuda")
            P_zero = torch.tensor(sorted(zero), device="cuda")
            model(input_ids=torch.tensor([a], device="cuda"), use_cache=False, logits_to_keep=1)
            plain = {L: (st[("k", L)][torch.cat([P_al, P_zero])].half().cpu(), st[("y", L)][torch.cat([P_al, P_zero])].float().pow(2).mean(-1).sqrt().cpu(),
                         st[("h", L)][P_al].float().cpu()) for L in LAY}
            for tag in ("ins", "ctl"):
                shift = pos[tag][2]
                model(input_ids=torch.tensor([seqs[tag]], device="cuda"), use_cache=False, logits_to_keep=1)
                for L in LAY:
                    d = st[("h", L)][P_al + shift].float().cpu() - plain[L][2]
                    rec[L]["D"][tag].append(torch.cat([d, torch.zeros(len(P_zero), d.shape[1])]).half())
            for L in LAY:
                rec[L]["K"].append(plain[L][0])
                rec[L]["Y"].append(plain[L][1])
            n_used += 1
            if n_used % 40 == 0:
                log(f"[i2n] state {n_used}")
            if n_used >= N_STATES:
                break
    for h in hooks:
        h.remove()
    W = {L: dict(wp=Ls[L].post_feedforward_layernorm.weight.float().cpu(), s=float(Ls[L].layer_scalar.float()), Wn=float(Ls[L].mlp.down_proj.weight.float().norm()))
         for L in LAY}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    rep = dict(states=n_used, rows_per_layer=int(sum(len(x) for x in rec[LAY[0]]["K"])), layers={})
    arms = {}
    for L in LAY:
        K = torch.cat(rec[L]["K"]).double().cuda()
        rms = torch.cat(rec[L]["Y"]).double().cuda()
        G = K.T @ K
        lam = 1e-2 * float(torch.diagonal(G).mean())
        G += lam * torch.eye(G.shape[0], device="cuda", dtype=G.dtype)
        Gc = torch.linalg.cholesky(G)
        for tag in ("ins", "ctl"):
            D = torch.cat(rec[L]["D"][tag]).double().cuda()
            dY = (D / W[L]["s"]) * inv_scale(W[L]["wp"]).double().cuda()[None, :] * rms[:, None]
            dW = torch.cholesky_solve(K.T @ dY, Gc).T  # 1536 x d_ff
            fitted = K @ dW.T
            r2 = float(1 - ((fitted - dY) ** 2).sum() / ((dY - dY.mean(0)) ** 2).sum())
            U, S, Vh = torch.linalg.svd(dW.float(), full_matrices=False)
            share = float(S[:RANK].pow(2).sum() / S.pow(2).sum())
            rep["layers"][f"{tag}_L{L}"] = dict(target_norm_mean=round(float(D.norm(dim=1).mean()), 4), fit_r2=round(r2, 4), rank_energy_share=round(share, 4),
                                                delta_rel_to_W=round(float(dW.norm()) / W[L]["Wn"], 5))
            log(f"{tag} layer {L}", rep["layers"][f"{tag}_L{L}"])
            for name, t2, L2, alpha in ARMS:
                if t2 == tag and L2 == L:
                    arms[name] = save_lora(os.path.join(LAB_DIR, name), L, Vh[:RANK].cpu(), (U[:, :RANK] * S[:RANK] * alpha).cpu())
            del D, dY, dW, fitted
        del K, G, Gc
        torch.cuda.empty_cache()
    log("adapters", sorted(arms))
    json.dump(dict(rep=rep, arms=arms), open(FIT_OUT, "w"))


if __name__ == "__main__" and os.environ.get("EXP104_STAGE") == "fit":
    fit_stage()
elif __name__ == "__main__":
    with Experiment(EXP, "instruction-to-neurons", category="improvement", depends_on=["EXP-092", "EXP-102"],
                    motivation="The one lab result outside the noise floor is an instruction (D-098, D-099); its effect on E2B's internal states, "
                               "written into one MLP layer in closed form, would carry the behaviour in the weights.",
                    hypothesis="A down_proj edit fitted to reproduce the instruction's state change raises applying edits on unseen bench states "
                               "above the unedited arm's noise range, approaching the instruction itself, and beyond the same edit fitted to a neutral instruction.",
                    config=dict(layers=LAY, rank=RANK, states=N_STATES, instruction=INSTR, control_instruction=CTRL, arms=[a[0] for a in ARMS])) as ex:
        import shutil
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        flog = f"/content/{EXP}_fit.log"
        with open(flog, "w") as fl:
            fit = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, EXP104_STAGE="fit"), stdout=fl, stderr=subprocess.STDOUT)
        ex.log("fit stage exit", fit.returncode, "in", round(time.time() - t0), "s; log tail:", open(flog).read()[-3000:])
        F = json.load(open(FIT_OUT))
        rep, arms = F["rep"], F["arms"]
        ex.save_json("fit.json", rep)
        os.makedirs("/content/fastlab/jobs", exist_ok=True)
        job = "x104"
        json.dump(dict(name=job, split="quick", arms=arms, variants={"instruction": INSTR}), open(f"/content/fastlab/jobs/{job}.json", "w"))
        t0 = time.time()
        srv = subprocess.run(["/opt/sne/bin/python", os.path.join(HERE, "fastlab_server.py")], env=dict(os.environ, FASTLAB_ONCE="1"))
        ex.log("lab exit", srv.returncode, "in", round(time.time() - t0), "s")
        resf = f"/content/fastlab/results/{job}.json"
        for a, p in arms.items():  # adapters are small (rank 128, one module): keep all of them
            shutil.copytree(p, os.path.join(OUT, a), dirs_exist_ok=True)
        if not os.path.exists(resf):
            raise RuntimeError("the lab produced no result; adapters copied to " + OUT)
        res = json.load(open(resf))
        ex.save_json("bench_quick.json", res)
        order, base = res["order"], res["arms"]["base"]
        summ = {}
        for arm, v in res["arms"].items():
            row = {k: sum(bool(v[i].get(k)) for i in order) for k in ("applies", "correct", "esc", "selfcopy")}
            if arm != "base":
                for k in ("applies", "correct"):
                    x = sum(1 for i in order if v[i].get(k) and not base[i].get(k))
                    y = sum(1 for i in order if base[i].get(k) and not v[i].get(k))
                    row[k + "_gain_loss_p"] = (x, y, round(mcnemar(x, y), 3))
            summ[arm] = row
            ex.log(arm, row)
        ex.result(fit=rep, bench=summ, headline="instruction-to-neurons, quick bench (96 states): applies/correct " +
                  ", ".join(f"{a} {summ[a]['applies']}/{summ[a]['correct']}" for a in summ))

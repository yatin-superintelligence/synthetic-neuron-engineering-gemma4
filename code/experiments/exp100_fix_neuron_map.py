"""EXP-100: a map of E2B's MLP neurons by what they do when E2B writes a fix, and edits built from it (training-free).

The question (the author's, 2026-11-09): which neurons contribute to writing a correct fix, the way brain regions are tied to tasks, and
can the model be made much better by working on them? Data: E2B's own fix attempts on EXP-092's fit states (the 1,028 sampled answers
at temperature 1.0 and the 257 greedy ones), each labelled correct or not by the bench (the answer's edit equals the reference fix).
For each answer, teacher-forced in bf16, every MLP neuron's activation (the input to down_proj, all 35 layers, 6,144 or 12,288
neurons per layer, about 350,000 in all) is averaged over the tool-call tokens. Per neuron: Welch's t of correct against incorrect
answers. Reliability: the fit states are split in two halves (seed 100) and t is computed on each; the correlation of the two halves'
t over the top neurons is reported. The map is the count of strong neurons per layer.

Edits (exact LoRAs on down_proj: delta = (beta - 1) W_down[:, idx] E_idx, so the chosen neurons' writes are scaled by beta; at most
128 neurons per layer, so at most rank 128, the competition's limit):
  fix_top64_b15, fix_top64_b25, fix_top256_b15, fix_top256_b25: the neurons with the largest positive t, amplified;
  wrong_top256_b05: the neurons with the most negative t, halved;
  rnd256_b25, rnd256_b05: random neurons with the same per-layer counts as fix_top256, scaled the same ways (controls).
They are screened in the fast lab (D-082) against the unedited model; any edit that leads goes to the 257 held-out states and then to
an official agent run.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

HERE = os.path.dirname(__file__)
_src = open(os.path.join(HERE, "exp092_patch_neurons.py"), encoding="utf-8").read()
P = {"__file__": os.path.join(HERE, "exp092_patch_neurons.py"), "__name__": "exp092"}
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)
B92 = os.path.join(ROOT, "artifacts_local", "EXP-092")
OUT = os.path.join(ROOT, "artifacts_local", "EXP-100")
CAP = 128


def save_scaling(path, W, sel, beta):
    """sel: {layer: [neuron indices]}; W: {layer: down_proj weight (d x d_ff)}."""
    from safetensors.torch import save_file
    r = max(len(v) for v in sel.values() if v)
    sd = {}
    for L, idx in sel.items():
        if not idx:
            continue
        d, dff = W[L].shape
        A = torch.zeros(r, dff)
        B = torch.zeros(d, r)
        for k, j in enumerate(idx):
            A[k, j] = 1.0
            B[:, k] = (beta - 1.0) * W[L][:, j]
        base = f"base_model.model.model.language_model.layers.{L}.mlp.down_proj"
        sd[f"{base}.lora_A.weight"] = A.contiguous()
        sd[f"{base}.lora_B.weight"] = B.contiguous()
    os.makedirs(path, exist_ok=True)
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=r, lora_alpha=r, target_modules=["down_proj"],
                   layers_to_transform=sorted(L for L, v in sel.items() if v), lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True,
                   inference_mode=True, use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


with Experiment("EXP-100", "fix-neuron-map", category="localization", depends_on=["EXP-092"],
                motivation="Map which of E2B's MLP neurons are more active when it writes a correct fix than a wrong one, and edit the model through them.",
                hypothesis="Amplifying the neurons most tied to correct fixes (or damping those tied to wrong ones) raises correct bench answers by at least "
                           "half against the unedited model, beyond random neurons scaled the same way.",
                config=dict(cap_per_layer=CAP, edits=["fix_top64_b15", "fix_top64_b25", "fix_top256_b15", "fix_top256_b25", "wrong_top256_b05",
                                                      "rnd256_b25", "rnd256_b05"])) as ex:
    os.makedirs(OUT, exist_ok=True)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from sne.model import layers, load
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:24]
    states, _ = P["build_states"](tasks, ex)
    by_id = {s["iid"]: s for s in states}
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    saved = json.load(open(os.path.join(B92, "fit_samples.json"), encoding="utf-8"))
    greedy = json.load(open(os.path.join(B92, "base_greedy.json"), encoding="utf-8"))
    rng = random.Random(100)
    ids = sorted({s["iid"] for s in states})
    r92 = random.Random(92)
    r92.shuffle(ids)
    n = len(ids)
    fit = set(ids[: n // 2])
    answers = [(a["iid"], a["text"], bool(a["correct"])) for a in saved if a["iid"] in by_id and a["iid"] in fit]
    answers += [(i, v["text"], bool(v["correct"])) for i, v in greedy.items() if i in by_id and i in fit and v.get("text")]
    answers = [x for x in answers if "<|tool_call>" in x[1]]
    good = [x for x in answers if x[2]]
    bad = [x for x in answers if not x[2]]
    rng.shuffle(bad)
    use = good + bad[: (12 if SMOKE else max(400, 4 * len(good)))]
    ex.log("answers with a tool call", len(answers), "| correct", len(good), "| used", len(use))
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False)
    Ls = layers(model)
    nL = len(Ls)
    st, C0 = {}, [0]
    # each hook keeps only the mean over the tool-call tokens, so 35 layers of activations never sit in GPU memory at once
    hooks = [Ls[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: st.__setitem__(L, a[0][0][C0[0] - 1:].float().mean(0).half().cpu())) for L in range(nL)]
    feats = {L: [] for L in range(nL)}
    labels, sids = [], []
    with torch.no_grad():
        for k, (iid, text, ok) in enumerate(use):
            prompt = P["G"]["render"](tok, by_id[iid]["prefix"], True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(text, add_special_tokens=False)["input_ids"]
            c0 = len(p_ids) + len(tok(text[: text.find("<|tool_call>")], add_special_tokens=False)["input_ids"])
            C0[0] = c0
            model(input_ids=torch.tensor([p_ids + a_ids], device="cuda"), use_cache=False, logits_to_keep=1)
            for L in range(nL):
                feats[L].append(st[L])
            labels.append(ok)
            sids.append(iid)
            if k % 50 == 0:
                print(f"[map] answer {k}/{len(use)}", flush=True)
    for h in hooks:
        h.remove()
    W = {L: Ls[L].mlp.down_proj.weight.float().cpu() for L in range(nL)}
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    y = torch.tensor(labels)
    rh = random.Random(100)
    half = {s: int(rh.random() < 0.5) for s in sorted(set(sids))}
    hA = torch.tensor([half[s] == 0 for s in sids])

    def welch(X, m):
        a, b = X[m], X[~m]
        return (a.mean(0) - b.mean(0)) / torch.sqrt(a.var(0) / max(1, len(a)) + b.var(0) / max(1, len(b)) + 1e-12)
    T, TA, TB = {}, {}, {}
    for L in range(nL):
        X = torch.stack(feats[L]).float()
        T[L] = welch(X, y)
        TA[L] = welch(X[hA], y[hA])
        TB[L] = welch(X[~hA], y[~hA])
    allt = torch.cat([T[L] for L in range(nL)])
    allA = torch.cat([TA[L] for L in range(nL)])
    allB = torch.cat([TB[L] for L in range(nL)])
    top = torch.topk(allt.abs(), 2000).indices
    rel = float(torch.corrcoef(torch.stack([allA[top], allB[top]]))[0, 1])
    offs = []
    o = 0
    for L in range(nL):
        offs.append((o, L))
        o += len(T[L])

    def where(gi):
        for start, L in reversed(offs):
            if gi >= start:
                return L, gi - start

    def pick(scores, k):
        order = torch.argsort(scores, descending=True)
        sel, per = {L: [] for L in range(nL)}, {}
        for gi in order.tolist():
            L, j = where(gi)
            if len(sel[L]) < CAP:
                sel[L].append(j)
                if sum(len(v) for v in sel.values()) == k:
                    break
        return sel
    fix64, fix256 = pick(allt, 64), pick(allt, 256)
    wrong256 = pick(-allt, 256)
    rnd = {L: random.Random(1000 + L).sample(range(len(T[L])), len(fix256[L])) for L in range(nL)}
    strong = {L: int((T[L].abs() > 4).sum()) for L in range(nL)}
    rep = dict(answers=len(use), correct=len(good), split_half_correlation_top2000=round(rel, 4), strong_neurons_per_layer_abs_t_over_4=strong,
               fix256_per_layer={L: len(v) for L, v in fix256.items() if v}, wrong256_per_layer={L: len(v) for L, v in wrong256.items() if v},
               top10=[dict(layer=where(gi)[0], neuron=where(gi)[1], t=round(float(allt[gi]), 2)) for gi in torch.topk(allt, 10).indices.tolist()],
               bottom10=[dict(layer=where(gi)[0], neuron=where(gi)[1], t=round(float(allt[gi]), 2)) for gi in torch.topk(-allt, 10).indices.tolist()])
    ex.log("map:", json.dumps(rep))
    ex.save_json("map.json", rep)
    for name, sel, beta in [("fix_top64_b15", fix64, 1.5), ("fix_top64_b25", fix64, 2.5), ("fix_top256_b15", fix256, 1.5), ("fix_top256_b25", fix256, 2.5),
                            ("wrong_top256_b05", wrong256, 0.5), ("rnd256_b25", rnd, 2.5), ("rnd256_b05", rnd, 0.5)]:
        save_scaling(os.path.join(OUT, name), W, sel, beta)
    ex.result(map=rep, adapters=sorted(os.listdir(OUT)),
              headline=f"fix-neuron map: {len(good)} correct of {len(use)} answers; split-half t correlation {rel:.2f} over the top 2,000 neurons; "
                       f"strong neurons by layer {strong}")

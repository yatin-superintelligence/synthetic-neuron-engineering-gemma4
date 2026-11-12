"""EXP-101: data-to-neuron conversion. The fit tasks' correct fixes written into one MLP layer in closed form, tested on unseen tasks.

The author's question (2026-11-09): take the tasks as they are and convert them into neurons or a layer injected into the model. An MLP
layer acts as a key-value memory (its input is the key, its write the value), and such a memory can be filled in closed form from data,
without gradient training (the idea behind ROME and MEMIT). Here, at layer L of E2B:
  keys:   the down_proj input at each position that predicts a token of the reference edit (old_string and new_string of the correct
          edit_file call), teacher-forced in bf16 on the 257 fit states of EXP-092's patch bench;
  values: where E2B's own top token there is wrong, the down_proj output change that raises the reference token over E2B's choice by c
          logits to first order (carried back through the final norm and the post-feedforward norm, as sne.conditional.write_vector);
          where E2B is already right, no change;
  guard:  as many positions from the thought and the prompt with a zero target, so the memory stays quiet elsewhere;
  solve:  ridge least squares for delta W (one solve, seconds), then its best rank-64 approximation as an exact LoRA on down_proj.
Arms: layers 12 and 15 at c = 2 and 4; controls: the same solve with the values shuffled across positions (layer 12 and 15, c = 4), so
the memory has the same size and inputs but stores no task knowledge. Screened in the fast lab (D-082) on held-out states the fit never
saw; a leader goes to all 257 held-out states and then to official agent runs.
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

HERE = os.path.dirname(__file__)
_src = open(os.path.join(HERE, "exp092_patch_neurons.py"), encoding="utf-8").read()
P = {"__file__": os.path.join(HERE, "exp092_patch_neurons.py"), "__name__": "exp092"}
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)
OUT = os.path.join(ROOT, "artifacts_local", "EXP-101")
LAYERS = [12, 15]
CS = [2.0, 4.0]
RANK = 64
STR = re.compile(r'(?:old_string|new_string):<\|"\|>(.*?)<\|"\|>', re.S)


def gold_message(st):
    return {"role": "assistant", "content": "", "reasoning": "This matches the failing test. I will make this one-line fix.",
            "tool_calls": [{"type": "function", "function": {"name": "edit_file", "arguments": st["gold"]}}]}


def save_lora(path, L, A, B):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    base = f"base_model.model.model.language_model.layers.{L}.mlp.down_proj"
    save_file({f"{base}.lora_A.weight": A.contiguous(), f"{base}.lora_B.weight": B.contiguous()}, os.path.join(path, "adapter_model.safetensors"))
    r = A.shape[0]
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=r, lora_alpha=r, target_modules=["down_proj"],
                   layers_to_transform=[L], lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
                   use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


with Experiment("EXP-101", "data-to-neurons", category="improvement", depends_on=["EXP-092", "EXP-100"],
                motivation="The author's data-to-neuron idea: the fit tasks' correct fixes stored in one MLP layer by a closed-form least-squares memory.",
                hypothesis="A closed-form memory of the fit tasks' correct edits raises correct answers on held-out bench states against the unedited model "
                           "and against the same memory built from shuffled values.",
                config=dict(layers=LAYERS, c=CS, rank=RANK, control="values shuffled across positions")) as ex:
    os.makedirs(OUT, exist_ok=True)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from sne.conditional import text_model
    from sne.model import layers, load
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:24]
    states, _ = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    ids = sorted({s["iid"] for s in states})
    r92 = random.Random(92)
    r92.shuffle(ids)
    fit = set(ids[: len(ids) // 2])
    fit_states = [s for s in states if s["iid"] in fit]
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False)
    Ls, tm = layers(model), text_model(model)
    Wu = model.get_output_embeddings().weight
    E = Wu.float()
    wf = tm.norm.weight.float()
    st, C = {}, {}
    hooks = [tm.norm.register_forward_hook(lambda m, a, o: st.__setitem__("hf", o[0])),
             tm.norm.register_forward_pre_hook(lambda m, a: st.__setitem__("hpre", a[0][0]))]
    for L in LAYERS:
        hooks.append(Ls[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: st.__setitem__(("k", L), a[0][0])))
        hooks.append(Ls[L].mlp.register_forward_hook(lambda m, a, o, L=L: st.__setitem__(("y", L), o[0])))
    data = {L: dict(K=[], V=[], Kneg=[]) for L in LAYERS}
    rng = random.Random(101)
    n_pos = n_wrong = 0
    with torch.no_grad():
        for k, s in enumerate(fit_states):
            pre = P["G"]["render"](tok, s["prefix"], True)
            full = P["G"]["render"](tok, s["prefix"] + [gold_message(s)], False)
            if not full.startswith(pre):
                continue
            ans = full[len(pre):]
            p_ids = tok(pre, add_special_tokens=False)["input_ids"]
            enc = tok(ans, add_special_tokens=False, return_offsets_mapping=True)
            a_ids, offs = enc["input_ids"], enc["offset_mapping"]
            spans = [m.span(1) for m in STR.finditer(ans)]
            tgt = [j for j, (a, b) in enumerate(offs) if b > a and any(x <= a and b <= y for x, y in spans)]
            if not tgt:
                continue
            ids_t = torch.tensor([p_ids + a_ids], device="cuda")
            model(input_ids=ids_t, use_cache=False, logits_to_keep=1)
            n = len(p_ids)
            pos = torch.tensor([n + j - 1 for j in tgt], device="cuda")
            gold = torch.tensor([a_ids[j] for j in tgt], device="cuda")
            z = st["hf"][pos].float() @ E.T
            top = z.argmax(-1)
            wrong = top != gold
            rms_h = st["hpre"][pos].float().pow(2).mean(-1).sqrt()
            negp = torch.tensor(rng.sample(range(1, n), min(len(tgt), n - 1)), device="cuda")
            for L in LAYERS:
                Kp = st[("k", L)][pos].float()
                rms_y = st[("y", L)][pos].float().pow(2).mean(-1).sqrt()
                w = Ls[L].post_feedforward_layernorm.weight.float()
                sc = float(Ls[L].layer_scalar.float())
                e = E[gold] - E[top]  # zero rows where E2B is already right
                denom = e.pow(2).sum(-1, keepdim=True).clamp_min(1e-6)
                V = (rms_y * rms_h)[:, None] / sc * (e / denom) / (w * wf)[None, :]
                V[~wrong] = 0.0
                data[L]["K"].append(Kp.half().cpu())
                data[L]["V"].append(V.half().cpu())
                data[L]["Kneg"].append(st[("k", L)][negp].float().half().cpu())
            n_pos += len(tgt)
            n_wrong += int(wrong.sum())
            if k % 40 == 0:
                print(f"[d2n] state {k}/{len(fit_states)} positions {n_pos} wrong {n_wrong}", flush=True)
    for h in hooks:
        h.remove()
    Wd = {L: Ls[L].mlp.down_proj.weight.float().cpu() for L in LAYERS}
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    rep = dict(fit_states=len(fit_states), target_positions=n_pos, positions_where_e2b_is_wrong=n_wrong, layers={})
    g = torch.Generator().manual_seed(101)
    for L in LAYERS:
        K = torch.cat(data[L]["K"] + data[L]["Kneg"]).float().cuda()
        V1 = torch.cat(data[L]["V"]).float()
        V = torch.cat([V1, torch.zeros(sum(len(x) for x in data[L]["Kneg"]), V1.shape[1])]).cuda()
        lam = float(K.pow(2).sum(1).mean()) * 0.1
        G = K @ K.T + lam * torch.eye(len(K), device="cuda")  # dual ridge: delta W = V^T (K K^T + lam I)^-1 K
        for tag, Vt in (("mem", V), ("shuf", V[torch.randperm(len(V), generator=g).cuda()])):
            Alpha = torch.linalg.solve(G, Vt)
            dW = (Alpha.T @ K).cpu()  # D_out x D_in
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            share = float(S[:RANK].pow(2).sum() / S.pow(2).sum())
            for c in CS:
                if tag == "shuf" and c != max(CS):
                    continue
                B = (U[:, :RANK] * S[:RANK]) * c  # values were built for a 1-logit shift
                A = Vh[:RANK]
                name = f"{tag}_L{L}_c{int(c)}"
                save_lora(os.path.join(OUT, name), L, A, B)
                rep["layers"][name] = dict(rank_energy_share=round(share, 4), delta_norm=round(float(dW.norm() * c), 3),
                                           relative_to_W=round(float(dW.norm() * c / Wd[L].norm()), 5))
                ex.log(name, rep["layers"][name])
        del G
        torch.cuda.empty_cache()
    ex.save_json("memory.json", rep)
    ex.result(memory=rep, adapters=sorted(os.listdir(OUT)),
              headline=f"data-to-neuron memory from {len(fit_states)} fit tasks: {n_pos} target positions, E2B wrong at {n_wrong}; adapters "
                       f"{sorted(os.listdir(OUT))}")

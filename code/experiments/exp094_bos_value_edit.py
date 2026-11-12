"""EXP-094: deleting the attention-sink value of E2B's BOS token as an exact rank-1 weight edit, training-free.

Lead: the author's earlier neuron-editing study (private repo gemma-neuron-engineering, N60, 2026-11-02) deleted the BOS value row in
every attention call. On 16 development code-comprehension answer choices it scored 11 of 16 against 8 for both the unedited model
and a one-position control. Its fresh confirmation (N61) never completed. That edit was a runtime hook.
Here it is an exact weight edit, so it ships as a LoRA. BOS is always at position 0 and attends only to itself, so its hidden state,
and the normed input x_bos to each layer's v_proj, are the same in every prompt. W_v' = W_v - (W_v x^)(x^)^T with x^ = x_bos/|x_bos|
makes BOS's value exactly 0 at that layer (v_norm(0) = 0) and changes another token's value only by its component along x^. Layers 0
to 14 compute values. Layers 15 to 34 reuse those of layers 13 and 14, so the deletion covers all 35 attention calls, as N60's did.

Pre-registered (D-066, before any result), one fixed edit, no selection, so all items are used:
  Arms: the unedited model; bos (the rank-1 edit at layers 0 to 14); and control (the same rank-1 projection along a random unit
    direction in each layer's normed input space, seed 94; rank matched; its effect on ordinary tokens is reported beside BOS's).
  Measures, on the official W4A16 via vLLM with LoRA, greedy:
    MBPP+ (all 378 problems, up to 8,192 tokens);
    EXP-092's patch bench (all 514 states; correct and applies, scored as in EXP-092).
  Gate for an official agent run (seed 1, D-065): bos beats the unedited model on MBPP+ or on bench-correct with exact McNemar
    p < 0.05; it is not lower on the other measure; and it is above the control on the measure that passed.
  The edit's vectors come from the bf16 QAT checkpoint; on W4A16 the BOS value is then near 0 rather than exactly 0 (the residual
  is measured in bf16 and reported).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE, bench  # noqa: E402
from sne.registry import Experiment  # noqa: E402

_src = open(os.path.join(os.path.dirname(__file__), "exp092_patch_neurons.py"), encoding="utf-8").read()
P = {"__file__": os.path.join(os.path.dirname(__file__), "exp092_patch_neurons.py"), "__name__": "exp092"}
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)
OUT = os.path.join(ROOT, "artifacts_local", "EXP-094")
N_COMPUTE = 15


def vectors(ex):
    """x_bos per computing layer from a BOS-only forward of the bf16 QAT model, and the matching B = -W_v x^."""
    from huggingface_hub import snapshot_download
    from sne.model import layers, load
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, tok = load(hf, ple_on_cpu=False)
    Ls = layers(model)
    st = {}
    hooks = [Ls[i].self_attn.v_proj.register_forward_pre_hook(lambda m, a, i=i: st.__setitem__(i, a[0][0].float().cpu())) for i in range(N_COMPUTE)]
    with torch.no_grad():
        model(input_ids=torch.tensor([[tok.bos_token_id]], device="cuda"), use_cache=False, logits_to_keep=1)
        x0 = {i: st[i][0].clone() for i in range(N_COMPUTE)}
        # the same BOS state inside a real prompt (causality: identical up to numerics), and how much ordinary tokens point along x^
        prompt = "<|turn>user\nWrite a function to add two numbers.<turn|>\n<|turn>model\n"
        ids = [tok.bos_token_id] + tok(prompt, add_special_tokens=False)["input_ids"]
        model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()
    g = torch.Generator().manual_seed(94)
    out, rep = {}, {}
    for i in range(N_COMPUTE):
        xb = x0[i]
        xh = xb / xb.norm()
        W = Ls[i].self_attn.v_proj.weight.float().cpu()
        r = torch.randn(xh.shape, generator=g)
        rh = r / r.norm()
        toks = st[i][1:]
        cos_tok = (torch.nn.functional.normalize(toks, dim=-1) @ xh).abs()
        cos_rnd = (torch.nn.functional.normalize(toks, dim=-1) @ rh).abs()
        out[i] = dict(B=-(W @ xh), A=xh, Bc=-(W @ rh), Ac=rh)
        rep[i] = dict(bos_in_prompt_cos=round(float(torch.nn.functional.cosine_similarity(st[i][0], xb, dim=0)), 6),
                      residual_bos_value_bf16=round(float((W @ xb - (W @ xh) * (xh @ xb)).norm() / (W @ xb).norm()), 8),
                      tokens_mean_abs_cos_with_bos_dir=round(float(cos_tok.mean()), 4), tokens_mean_abs_cos_with_random_dir=round(float(cos_rnd.mean()), 4))
    ex.log("BOS directions", json.dumps(rep))
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return out, rep


def save_v_adapter(path, vec, which):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    sd = {}
    for i, v in vec.items():
        base = f"base_model.model.model.language_model.layers.{i}.self_attn.v_proj"
        A, B = (v["A"], v["B"]) if which == "bos" else (v["Ac"], v["Bc"])
        sd[f"{base}.lora_A.weight"] = A.view(1, -1).float().contiguous()
        sd[f"{base}.lora_B.weight"] = B.view(-1, 1).float().contiguous()
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=1, lora_alpha=1, target_modules=["v_proj"],
                   layers_to_transform=list(vec), lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
                   use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


class Suite:
    """sne.evaluate.run_suite's server interface on top of EXP-092's vLLM server, for one served model name."""

    def __init__(self, srv, model):
        self.srv, self.model = srv, model

    def complete_many(self, prompts, max_tokens, lora_scales=None, sampling=None):
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time()

        def one(p):
            try:
                r = self.srv._req(dict(model=self.model, prompt=p, max_tokens=max_tokens, temperature=0.0))
                c = r["choices"][0]
                return dict(text=c["text"], n_gen=r["usage"]["completion_tokens"], n_prompt=r["usage"]["prompt_tokens"],
                            stop_type="limit" if c.get("finish_reason") == "length" else "stop")
            except Exception:  # noqa: BLE001
                return dict(text="", n_gen=0, n_prompt=0, stop_type="error")
        with ThreadPoolExecutor(128) as ex:
            gens = list(ex.map(one, prompts))
        n = sum(g["n_gen"] for g in gens)
        return gens, dict(wall_s=round(time.time() - t0, 1), gen_tokens=n)


with Experiment("EXP-094", "bos-value-edit", category="improvement", depends_on=["EXP-092"],
                motivation="The strongest lead of the author's earlier 77-study neuron-editing repo (N60, BOS value deletion) made exact and shippable as "
                           "a rank-1 v_proj edit, tested on full programs and on the agent's patch step.",
                hypothesis="Deleting BOS's attention value raises MBPP+ passes or bench-correct answers against the unedited model (exact McNemar p < 0.05) and "
                           "against a rank-matched random-direction control, without lowering the other measure.",
                config=dict(layers=list(range(N_COMPUTE)), control="random unit direction per layer, seed 94", mbpp="all 378, cap 8,192",
                            bench="EXP-092 states, all 514")) as ex:
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    from sne.evaluate import run_suite
    vec, rep = vectors(ex)
    import gc
    gc.collect()
    torch.cuda.empty_cache()  # the HF model must leave the GPU before vLLM starts
    ad = {"bos": save_v_adapter(os.path.join(OUT, "bos"), vec, "bos"), "control": save_v_adapter(os.path.join(OUT, "control"), vec, "control")}
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    mbpp = bench.mbppplus("dev") + bench.mbppplus("test")
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        mbpp, tasks = mbpp[:6], tasks[:12]
    states, _ = P["build_states"](tasks, ex)

    def vprompt(prefix):
        p = P["G"]["render"](tok, prefix, True)
        return p[len("<bos>"):] if p.startswith("<bos>") else p
    prompts = [vprompt(s["prefix"]) for s in states]
    srv = P["Server"](ad, os.path.join(OUT, "vllm.log"))
    res = {}
    try:
        for name in ("base", "bos", "control"):
            summ, rows = run_suite(Suite(srv, name), tok, mbpp, max_tokens=8192)
            res[name] = dict(mbpp={r["task_id"]: r["passed"] for r in rows}, mbpp_summary={k: summ[k] for k in ("passed", "n", "categories")})
            outs = srv.answers(name, prompts)
            sc = [P["score"](o[0], s) for s, o in zip(states, outs)]
            res[name]["bench_correct"] = [x["correct"] for x in sc]
            res[name]["bench_applies"] = [x["applies"] for x in sc]
            ex.log(f"[bos] {name}: MBPP+ {summ['passed']}/{summ['n']} | bench correct {sum(res[name]['bench_correct'])}/{len(sc)} "
                   f"applies {sum(res[name]['bench_applies'])}/{len(sc)}")
    finally:
        srv.close()
    ids = [t["task_id"] for t in mbpp]
    st = {}
    for arm in ("bos", "control"):
        st[arm] = dict(mbpp=P["mcnemar"]([res[arm]["mbpp"][i] for i in ids], [res["base"]["mbpp"][i] for i in ids]),
                       bench_correct=P["mcnemar"](res[arm]["bench_correct"], res["base"]["bench_correct"]),
                       bench_applies=P["mcnemar"](res[arm]["bench_applies"], res["base"]["bench_applies"]))
    b = st["bos"]
    tot = {arm: dict(mbpp=sum(res[arm]["mbpp"].values()), bench_correct=sum(res[arm]["bench_correct"]), bench_applies=sum(res[arm]["bench_applies"]))
           for arm in res}
    pass_mbpp = b["mbpp"][2] < 0.05 and b["mbpp"][0] > b["mbpp"][1] and tot["bos"]["bench_correct"] >= tot["base"]["bench_correct"] and tot["bos"]["mbpp"] > tot["control"]["mbpp"]
    pass_bench = (b["bench_correct"][2] < 0.05 and b["bench_correct"][0] > b["bench_correct"][1] and tot["bos"]["mbpp"] >= tot["base"]["mbpp"]
                  and tot["bos"]["bench_correct"] > tot["control"]["bench_correct"])
    ex.result(directions=rep, totals=tot, mcnemar={a: {k: dict(gained=v[0], lost=v[1], p=round(v[2], 4)) for k, v in d.items()} for a, d in st.items()},
              gate_passed=bool(pass_mbpp or pass_bench),
              headline=f"BOS value edit: MBPP+ {tot['bos']['mbpp']} vs {tot['base']['mbpp']} (control {tot['control']['mbpp']}) of {len(ids)}; bench correct "
                       f"{tot['bos']['bench_correct']} vs {tot['base']['bench_correct']} (control {tot['control']['bench_correct']}) of {len(states)}; gate "
                       f"{'passed' if (pass_mbpp or pass_bench) else 'not passed'}")

"""EXP-132 (D-158): the thinking-loop direction of Gemma 4 12B, found and removed in closed form, as EXP-040 and EXP-053 did for E2B.

Why: unedited 12B (EXP-014d, seed 1) solves 16 of the 40 reference-resolvable tasks. 26 of its 40 runs end in a context overflow and
6 of those are solved; of the 14 that do not overflow it solves 10, close to 31B on the same tasks (12). 33 of its 58 runs contain a
thinking loop (54 loops; the looping response has a median 4,267 tokens, at the thinking cap, with its top sentence repeated a median
36 times): a thought at the cap arrives when the prompt is near the compaction threshold and the next request exceeds the context.
The E2B loop edit cut loops in replays from 15 to 2 of 48 (paper Section 5) but could not raise tasks solved, because E2B's limit is
the content of the fix. 12B's limit on these tasks is the loop.

Method (EXP-040, unchanged): the loop bench is built from EXP-014d's traces (loop responses and control responses, the shared dev and
test split of tasks). Teacher-forced on the bf16 QAT checkpoint, for each dev case the last 3,000 prompt tokens and up to 1,500 response
tokens; token classes LOOP (after loop onset) and OTHER (before onset, and control responses). Per layer, on the raw MLP output,
v = normalise(mean LOOP - mean OTHER) fitted on half of the dev tasks; held-out separation d' on the other half, against 32 random
unit directions. Layers: the 4 with the highest held-out d' (EXP-053's rule). Edit: W' = (I - v v^T) W on down_proj of those layers,
an exact rank-1 LoRA per layer (B = -v, A = v^T W, W from the bf16 checkpoint). Control: the same edit along random unit directions
in the same layers (seed 132). The test half of the tasks is not used here.
Outputs (artifacts_local/EXP-132): loop_edit and loop_control (PEFT adapters), layers.json, directions.pt, loopbench_12b.jsonl.
Runs on a Colab G4 with --official (the bf16 12B is about 24 GB). EXP132_MODEL overrides the model (a local folder for a pipeline test).
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE, loops as L  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("EXP132_MODEL", "google/gemma-4-12B-it-qat-q4_0-unquantized")
TRACES = os.path.join(ROOT, "artifacts_local", "colab", "exp-014d", "results", "traces")
EXPID = os.environ.get("EXP132_ID", "EXP-132")
FIXED_LAYERS = [int(x) for x in os.environ.get("EXP132_LAYERS", "").split(",") if x]  # D-158a: layers named beforehand instead of the top 4 by d'
OUT = os.path.join(ROOT, "scratch", "preflight", EXPID) if SMOKE else os.path.join(ROOT, "artifacts_local", EXPID)
CTX, RESP, PER_CLASS, N_LAYERS_EDIT = 3000, 1500, 200, 4


def save_adapter(name, parts, base_name):
    """parts: {layer: (B [out, 1], A [1, in])}; lora_alpha = r = 1, so the scale is exactly 1."""
    from safetensors.torch import save_file
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    tensors, rank_pattern = {}, {}
    for i, (B, A) in sorted(parts.items()):
        key = f"base_model.model.model.language_model.layers.{i}.mlp.down_proj"
        tensors[f"{key}.lora_A.weight"] = A.float().contiguous()
        tensors[f"{key}.lora_B.weight"] = B.float().contiguous()
        rank_pattern[f"model.language_model.layers.{i}.mlp.down_proj"] = 1
    save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=base_name, r=1, lora_alpha=1, rank_pattern=rank_pattern,
               alpha_pattern={k: 1 for k in rank_pattern}, target_modules=["down_proj"], lora_dropout=0.0, bias="none",
               fan_in_fan_out=False, init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    return d


with Experiment(EXPID, "12b-loop-direction", category="improvement", depends_on=["EXP-014d", "EXP-040", "EXP-053"],
                motivation="26 of 12B's 40 runs on the reference-resolvable tasks end in a context overflow that follows a thinking loop; "
                           "where it does not overflow 12B solves 10 of 14. The loop direction found and removed in E2B is rebuilt for 12B.",
                hypothesis="A few 12B layers separate loop tokens on held-out tasks (d' well above random directions); the agent test "
                           "(EXP-133) measures loops, context overflows and tasks solved.",
                config=dict(model=REPO, ctx_tokens=CTX, resp_tokens=RESP, tokens_per_class=PER_CLASS, layers_edited=N_LAYERS_EDIT,
                            fit_half_seed=132, traces="EXP-014d", fixed_layers=FIXED_LAYERS)) as ex:
    os.makedirs(OUT, exist_ok=True)
    L.LOOPBENCH = os.path.join(OUT, "loopbench_12b.jsonl")  # never the E2B bench file
    rows = L.build({"exp-014d": TRACES}, n_controls=60)
    cases = [r for r in rows if r["split"] == "dev"]
    if SMOKE:
        cases = [c for c in cases if c["kind"] == "loop"][:3] + [c for c in cases if c["kind"] != "loop"][:3]
    tasks = sorted({c["instance_id"] for c in cases})
    random.Random(132).shuffle(tasks)
    fit_tasks = set(tasks[: len(tasks) // 2])
    ex.log("dev cases", len(cases), "| loop", sum(c["kind"] == "loop" for c in cases), "| tasks", len(tasks), "| fit tasks", len(fit_tasks))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    if os.path.isdir(REPO):
        path = REPO
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO, local_dir="/content/hf_12b")
    tok = AutoTokenizer.from_pretrained(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map=dev, low_cpu_mem_usage=True).eval()
    for nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
        if hasattr(model.model, nm) and getattr(model.model, nm) is not None:
            setattr(model.model, nm, None)
    Ls = model.model.language_model.layers
    nL = len(Ls)
    D = Ls[0].mlp.down_proj.out_features
    ex.log("model", REPO, "| layers", nL, "| hidden size", D, "| MLP width", Ls[0].mlp.down_proj.in_features)
    state = {}

    def mk(i):
        def hook(mod, args, out):
            state.setdefault("y", {})[i] = out[0, state["start"]:].float().cpu()
        return hook
    hs = [Ls[i].mlp.register_forward_hook(mk(i)) for i in range(nL)]
    rng = np.random.default_rng(132)
    store = {p: {i: {"LOOP": [], "OTHER": []} for i in range(nL)} for p in ("fit", "hold")}
    t0 = time.time()
    used = 0
    with torch.no_grad():
        for k, c in enumerate(cases):
            a = tok(L.render(tok, c), add_special_tokens=False)["input_ids"][-CTX:]
            text = c["original_text"]
            b = tok(text, add_special_tokens=False)["input_ids"][:RESP]
            if len(b) < 20:
                continue
            onset = len(tok(text[: c["onset_char"]], add_special_tokens=False)["input_ids"]) if c["kind"] == "loop" and c["onset_char"] else None
            state.clear()
            state["start"] = 1 + len(a)
            model(input_ids=torch.tensor([[tok.bos_token_id] + a + b], device=dev), use_cache=False, logits_to_keep=1)
            part = "fit" if c["instance_id"] in fit_tasks else "hold"
            pos = np.arange(len(b))
            loop_pos, other_pos = (pos[onset:], pos[:onset]) if onset is not None else (pos[:0], pos)
            loop_pos = rng.permutation(loop_pos)[:PER_CLASS]
            other_pos = rng.permutation(other_pos)[:PER_CLASS]
            for i in range(nL):
                y = state["y"][i]
                if len(loop_pos):
                    store[part][i]["LOOP"].append(y[loop_pos].clone())
                if len(other_pos):
                    store[part][i]["OTHER"].append(y[other_pos].clone())
            used += 1
            if k % 10 == 0:
                ex.log("case", k, f"{time.time() - t0:.0f} s")
    for h in hs:
        h.remove()
    ex.log("cases used", used)
    rows_l, vs = [], {}
    torch.manual_seed(132)
    for i in range(nL):
        s = store
        if not (s["fit"][i]["LOOP"] and s["fit"][i]["OTHER"] and s["hold"][i]["LOOP"] and s["hold"][i]["OTHER"]):
            continue
        fl, fo = torch.cat(s["fit"][i]["LOOP"]), torch.cat(s["fit"][i]["OTHER"])
        hl, ho = torch.cat(s["hold"][i]["LOOP"]), torch.cat(s["hold"][i]["OTHER"])
        v = fl.mean(0) - fo.mean(0)
        v = v / v.norm()

        def dp(u):
            x, y = hl @ u, ho @ u
            return float((x.mean() - y.mean()) / (0.5 * (x.var() + y.var())).sqrt().clamp_min(1e-8))
        rnd = [abs(dp(x / x.norm())) for x in torch.randn(32, D)]
        v2 = hl.mean(0) - ho.mean(0)
        rows_l.append(dict(layer=i, d_prime_heldout=round(dp(v), 3), random_abs_d_max=round(max(rnd), 3), random_abs_d_mean=round(float(np.mean(rnd)), 3),
                           cosine_between_halves=round(float((v @ v2) / v2.norm()), 3), n_loop_tokens=int(fl.shape[0] + hl.shape[0]),
                           n_other_tokens=int(fo.shape[0] + ho.shape[0])))
        vs[i] = v
    ex.save_json("layers.json", rows_l)
    torch.save(vs, os.path.join(OUT, "directions.pt"))
    top = sorted(rows_l, key=lambda r: -r["d_prime_heldout"])[:N_LAYERS_EDIT]
    layers_edit = sorted(r["layer"] for r in top)
    if FIXED_LAYERS:
        layers_edit = sorted(FIXED_LAYERS)
        top = [r for r in rows_l if r["layer"] in layers_edit]
    ex.log("layers chosen (highest held-out d')", [(r["layer"], r["d_prime_heldout"], r["random_abs_d_max"], r["cosine_between_halves"]) for r in top])
    base_name = "google/gemma-4-12B-it"
    edit, ctrl, overlap = {}, {}, {}
    for i in layers_edit:
        W = Ls[i].mlp.down_proj.weight.detach().float().cpu()  # [hidden, mlp width]
        v = vs[i]
        r = torch.nn.functional.normalize(torch.randn(D), dim=0)
        overlap[i] = round(float(r @ v), 4)
        edit[i] = (-v.view(-1, 1), (v.view(1, -1) @ W))
        ctrl[i] = (-r.view(-1, 1), (r.view(1, -1) @ W))
    # exactness of the edit on the bf16 weights: after W + B A the direction is gone from the layer's write
    chk = {i: round(float((edit[i][0] @ edit[i][1] + Ls[i].mlp.down_proj.weight.detach().float().cpu()).T.matmul(vs[i]).norm()), 6) for i in layers_edit}
    saved = [save_adapter("loop_edit", edit, base_name), save_adapter("loop_control", ctrl, base_name)]
    ex.result(layers=layers_edit, top=top, random_overlap_with_loop_direction=overlap, residual_write_along_direction_after_edit=chk,
              cases_used=used, adapters=[os.path.basename(s) for s in saved],
              headline=f"12B loop direction: layers {layers_edit}, held-out d' " + ", ".join(f"L{r['layer']} {r['d_prime_heldout']} (random max {r['random_abs_d_max']})" for r in top))

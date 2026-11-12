"""EXP-135 (D-160): a synthetic loop-breaker neuron for Gemma 4 12B, built in closed form (EXP-051's design, never completed on E2B).

Why: 26 of 12B's 40 runs on the reference-resolvable tasks end in a context overflow after a thinking loop (D-158); the looping thought
repeats one sentence a median of 36 times until the 4,096-token thinking cap. Removing a direction (EXP-132b, EXP-133a) acts on how the
loop is represented; this neuron acts on the decision the model fails to take: to end the thought.

The neuron: one new unit in the last layer's MLP. Read-out v over the neuron activations entering down_proj, fitted by weighted ridge
to be about 1 at sentence ends after a loop's onset (the model has just repeated a sentence it has already written) and about 0 at
sentence ends before the onset, at sentence ends of normal thoughts, and at other tokens. Write u: raises the end-of-thought token over
the tokens most probable at those positions by c logits where the read-out is 1 (through the final norm and the unembedding).
D' = D + u v^T, an exact rank-1 LoRA.
Data: the 12B loop bench of EXP-132 (EXP-014d's traces), dev half only: 22 loop responses and 27 control responses; teacher-forced on
the bf16 QAT checkpoint, the last 3,000 prompt tokens and up to 3,000 response tokens. Ridge strength from (0.001, 0.01, 0.1, 1, 10),
chosen on a held-out third of the dev tasks: the largest held-out mean at loop sentence ends with the 99th percentile at normal
sentence ends under 0.1 (EXP-051's rule); if no value passes, no neuron is built. Write strength c: the median gap between the most
probable token and the end-of-thought token at the fitting positives plus 1, clipped to 3 to 9. Control: the same write with a random
read-out of the same norm (seed 135). Outputs (artifacts_local/EXP-135): breaker and breaker_control (PEFT adapters), fit.json.
Runs on a Colab G4 with --official (the bf16 12B is about 24 GB).
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE, conditional as C, loops as L  # noqa: E402
from sne.model import layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("EXP135_MODEL", "google/gemma-4-12B-it-qat-q4_0-unquantized")
TRACES = os.path.join(ROOT, "artifacts_local", "colab", "exp-014d", "results", "traces")
OUT = os.path.join(ROOT, "scratch", "preflight", "EXP-135") if SMOKE else os.path.join(ROOT, "artifacts_local", "EXP-135")
CTX, RESP = 3000, 3000
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)


def save_adapter(name, layer, u, v, base_name):
    from safetensors.torch import save_file
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    key = f"base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
    save_file({f"{key}.lora_A.weight": v.view(1, -1).float().contiguous(), f"{key}.lora_B.weight": u.view(-1, 1).float().contiguous()},
              os.path.join(d, "adapter_model.safetensors"))
    rp = {f"model.language_model.layers.{layer}.mlp.down_proj": 1}
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=base_name, r=1, lora_alpha=1, rank_pattern=rp,
               alpha_pattern={k: 1 for k in rp}, target_modules=["down_proj"], lora_dropout=0.0, bias="none", fan_in_fan_out=False,
               init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    return d


with Experiment("EXP-135", "12b-loop-breaker-neuron", category="improvement", depends_on=["EXP-014d", "EXP-051", "EXP-132"],
                motivation="12B's looping thought runs to the thinking cap and the run dies of a context overflow. A neuron that reads "
                           "'a sentence was just repeated' and writes the end-of-thought token ends the thought at the loop's start.",
                hypothesis="The read-out separates loop sentence ends from normal sentence ends on held-out tasks (99th percentile at "
                           "normal sentence ends under 0.1); the agent test measures loops, overflows and tasks solved.",
                config=dict(model=REPO, ctx_tokens=CTX, resp_tokens=RESP, lambdas=LAMBDAS, split="dev", held_out_fraction="1/3 of dev tasks, seed 135",
                            selection_rule="max held-out pos_mean with hard-negative p99 < 0.1")) as ex:
    os.makedirs(OUT, exist_ok=True)
    L.LOOPBENCH = os.path.join(OUT, "loopbench_12b.jsonl")  # never the E2B bench file
    rows = L.build({"exp-014d": TRACES}, n_controls=60)
    cases = [r for r in rows if r["split"] == "dev"]
    if SMOKE:
        cases = [c for c in cases if c["kind"] == "loop"][:2] + [c for c in cases if c["kind"] != "loop"][:2]
    tasks = sorted({c["instance_id"] for c in cases})
    random.Random(135).shuffle(tasks)
    held = set(tasks[: max(1, len(tasks) // 3)])
    ex.log("dev cases", len(cases), "| loop", sum(c["kind"] == "loop" for c in cases), "| tasks", len(tasks), "| held-out tasks", len(held))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sne.chunked_attention import install
    install()
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
    layer = len(layers(model)) - 1
    ex.log("model", REPO, "| layers", layer + 1, "| neurons entering down_proj", layers(model)[layer].mlp.down_proj.in_features)
    with torch.no_grad():
        d = C.gather_loops(model, tok, cases, layer=layer, ctx=CTX, resp=RESP, per_case_neg=200, seed=135, dev=dev)
    gaps = sorted(d["gap"])
    ex.log("positives (loop sentence ends)", len(d["pos"]), "| negatives", len(d["neg"]), "| of them normal sentence ends", len(d["hard"]),
           "| gap from the most probable token to end-of-thought at positives: median", round(float(np.median(gaps)), 2),
           "10th to 90th percentile", round(gaps[len(gaps) // 10], 2), "to", round(gaps[(9 * len(gaps)) // 10], 2))
    pm = torch.tensor([t in held for t in d["pos_task"]])
    nm_ = torch.tensor([t in held for t in d["neg_task"]])
    hm = torch.tensor([t in held for t in d["hard_task"]])
    grid = []
    for lam in LAMBDAS:
        v = C.fit_readout(d["pos"][~pm].float(), d["neg"][~nm_].float(), lam=lam)
        grid.append(dict(lam=lam, all_neg=C.separation(v, d["pos"][pm].float(), d["neg"][nm_].float()), hard=C.separation(v, d["pos"][pm].float(), d["hard"][hm].float())))
        ex.log(grid[-1])
    ok = [g for g in grid if g["hard"]["neg_p99"] < 0.1]
    gap = float(np.median(gaps))
    c = float(np.clip(gap + 1, 3, 9))
    out = dict(positives=len(d["pos"]), negatives=len(d["neg"]), hard=len(d["hard"]), held_positives=int(pm.sum()), held_hard=int(hm.sum()),
               grid=grid, median_gap=round(gap, 3), c=c, layer=layer)
    saved = []
    if ok or SMOKE:
        best = max(ok or grid, key=lambda g: g["hard"]["pos_mean"])
        v = C.fit_readout(d["pos"].float(), d["neg"].float(), lam=best["lam"])
        u = C.write_vector(model, layer, d["close_id"], d["top_ids"], c, float(np.mean(d["rms_y"])), float(np.mean(d["rms_h"])))
        torch.manual_seed(135)
        r = torch.randn_like(v)
        r = r / r.norm() * v.norm()
        out.update(chosen=best, v_norm=round(float(v.norm()), 4), u_norm=round(float(u.norm()), 4),
                   control_readout_at_positives=round(float((d["pos"].float() @ r).mean()), 4))
        base_name = "google/gemma-4-12B-it"
        saved = [save_adapter("breaker", layer, u, v, base_name), save_adapter("breaker_control", layer, u, r, base_name)]
    ex.save_json("fit.json", out)
    ex.result(fit={k: v for k, v in out.items() if k != "grid"}, adapters=[os.path.basename(s) for s in saved],
              headline=(f"12B loop-breaker neuron: held-out read-out at loop sentence ends {out['chosen']['hard']['pos_mean']}, "
                        f"99th percentile at normal sentence ends {out['chosen']['hard']['neg_p99']}, gap {round(gap, 2)}, c {c}"
                        if saved else f"no ridge strength passed the selection rule (gap {round(gap, 2)}); no neuron built"))

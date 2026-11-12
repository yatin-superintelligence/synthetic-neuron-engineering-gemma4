"""EXP-075 (build step): adapters that amplify the tool-call syntax neuron group, for testing on real tasks on Colab.

Why: in EXP-063 (32 saved edit moments, Q4_0, seed 0), switching off the 2,480 neurons of atlas groups 3, 12, 1 and 4 lowered the
share of edit calls that applied from 20 of 27 to 11 of 30, and "old_string not found" rose from 5 to 14. 2,480 matched random neurons
gave 16 of 26. So this group carries part of exact copying, which is the largest mechanical failure on real tasks (51 of 110 edit calls
failed in EXP-044a). The opposite move, amplifying it, may raise exact copying. Whether it does, and whether more tasks get solved, is
measured on the official stack, not on replays.

Arms, fixed before any run: amplify the group by 1.25 and by 1.5, and amplify the matched random neurons of EXP-063 (same seed 63, same
count per layer) by 1.5. The edit scales down_proj columns, an exact LoRA of rank equal to the neurons per layer (at most 452, within the
harness's 512 bucket). Only the PEFT form is built: the official stack loads PEFT adapters, and the F16 QAT checkpoint is the source.
The weights are stored at float16, which halves the upload.
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT  # noqa: E402
from sne.artifacts import LOCAL, base_weights  # noqa: E402
from sne.lora import Edit, neuron_scale  # noqa: E402

D62 = glob.glob(os.path.join(ROOT, "experiments", "EXP-062_*"))[0]
GROUPS = (3, 12, 1, 4)
WIDTHS = [6144] * 15 + [12288] * 20


def sets():
    groups = {g["group"]: g for g in json.load(open(os.path.join(D62, "groups.json"), encoding="utf-8"))}
    sel = [tuple(n) for g in GROUPS for n in groups[g]["neurons"]]
    rng = random.Random(63)  # the same draw as EXP-063's rand_syntax_on
    taken, counts = set(sel), {}
    for layer, _ in sel:
        counts[layer] = counts.get(layer, 0) + 1
    rand = [(layer, j) for layer, c in counts.items() for j in rng.sample([x for x in range(WIDTHS[layer]) if (layer, x) not in taken], c)]
    return sel, rand


def build(name, sel, alpha):
    e = Edit(f"EXP-075:{name}:f16")
    bw = base_weights("f16")
    by = {}
    for layer, j in sel:
        by.setdefault(layer, []).append(j)
    for layer, idx in sorted(by.items()):
        neuron_scale(e, bw.get(layer, "mlp.down_proj"), layer, idx, alpha)
    e.meta.update(n_neurons=len(sel), alpha=alpha, groups=list(GROUPS))
    # Every module is padded with zero rows to the same rank, and lora_alpha equals that rank, so the scale is exactly 1 in any loader.
    # Layers differ in rank here (1 to 452); a loader that ignores per-module rank settings would otherwise mis-scale them silently.
    from safetensors.torch import save_file
    rmax = max(A.shape[0] for B, A in e.deltas.values())
    tensors = {}
    for (i, mod), (B, A) in sorted(e.deltas.items()):
        r = A.shape[0]
        Ap = torch.zeros(rmax, A.shape[1])
        Ap[:r] = A
        Bp = torch.zeros(B.shape[0], rmax)
        Bp[:, :r] = B
        base = f"base_model.model.model.language_model.layers.{i}.{mod}"
        tensors[f"{base}.lora_A.weight"] = Ap.to(torch.float16).contiguous()
        tensors[f"{base}.lora_B.weight"] = Bp.to(torch.float16).contiguous()
    d = os.path.join(LOCAL, "EXP-075", name)
    os.makedirs(d, exist_ok=True)
    st = os.path.join(d, "adapter_model.safetensors")
    save_file(tensors, st)
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=rmax, lora_alpha=rmax,
               target_modules=["down_proj"], lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
               use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    # check: the stored adapter reproduces the intended change exactly, on the largest layer
    (li, mod), (B, A) = max(e.deltas.items(), key=lambda kv: kv[1][1].shape[0])
    k = f"base_model.model.model.language_model.layers.{li}.{mod}"
    got = tensors[k + ".lora_B.weight"].float() @ tensors[k + ".lora_A.weight"].float() * (cfg["lora_alpha"] / cfg["r"])
    err = float((got - B @ A).abs().max() / (B @ A).abs().max())
    assert err < 2e-3, err
    meta = dict(name=name, n_modules=len(e.deltas), rank=rmax, n_params_nonzero=e.n_params(), max_rank=max(len(v) for v in by.values()),
                bytes=os.path.getsize(st), check_layer=li, relative_error_fp16=round(err, 6), meta=e.meta)
    os.makedirs(os.path.join(ROOT, "experiments", "EXP-075_syntax-amplify-real-tasks"), exist_ok=True)
    json.dump(meta, open(os.path.join(ROOT, "experiments", "EXP-075_syntax-amplify-real-tasks", f"edit_{name}.json"), "w", encoding="utf-8"), indent=1, default=str)
    print(name, meta["n_modules"], "modules, max rank", meta["max_rank"], round(meta["bytes"] / 1e6, 1), "MB")


if __name__ == "__main__":
    sel, rand = sets()
    assert len(sel) == len(rand) == 2480, (len(sel), len(rand))
    build("amp_syntax_a125", sel, 1.25)
    build("amp_syntax_a150", sel, 1.5)
    build("amp_rand_a150", rand, 1.5)

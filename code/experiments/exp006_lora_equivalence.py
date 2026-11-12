"""EXP-006: end-to-end check of the neuron-edit compiler.

One non-trivial edit (zero-ablation of the top-2048 code neurons, EXP-033's code_top2048) is realised three ways:
  (a) PyTorch: deltas added in place to the bf16 HF model (reference semantics of the edit),
  (b) llama.cpp F16 GGUF + LoRA adapter at scale 1 (the deployment path used in every experiment),
  (c) llama.cpp F16 GGUF + LoRA adapter at scale 0 (must reproduce the unedited base exactly).
Greedy continuations on the 40 EXP-003 prompts are compared. (b) vs (a) should agree about as well as the unedited
engines agree in EXP-003; (c) must be identical to the plain base server output.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import DATA, MODELS, ROOT  # noqa: E402
from sne.artifacts import base_weights  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.evaluate import render  # noqa: E402
from sne.lora import Edit, apply_torch, neuron_scale  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF  # noqa: E402

N, NEW = 40, 128
HF = os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized")
E33 = glob.glob(os.path.join(ROOT, "experiments", "EXP-033_*"))
with Experiment("EXP-006", "lora-path-equivalence", category="infrastructure", depends_on=["EXP-003", "EXP-033"],
                motivation="Every edit is evaluated through LoRA adapters in llama.cpp. This checks that the adapter "
                           "computes the intended edit (vs. a direct weight edit) and that scale 0 is a true no-op.",
                hypothesis="Adapter-at-scale-0 output is identical to the base. Adapter-at-scale-1 agrees with the "
                           "PyTorch in-place edit at the same level as unedited F16 vs HF bf16 in EXP-003.",
                config=dict(edit="zero-ablate EXP-031 code_top 2048", n_prompts=N, new_tokens=NEW)) as ex:
    rank = json.load(open(glob.glob(os.path.join(ROOT, "experiments", "EXP-031_*", "neuron_rankings.json"))[0]))
    sel = {}
    for L, j, _, _ in rank["code_top"][:2048]:
        sel.setdefault(L, []).append(j)
    e = Edit("exp006")
    bw = base_weights("f16")
    for L, idx in sorted(sel.items()):
        neuron_scale(e, bw.get(L, "mlp.down_proj"), L, idx, 0.0)
    d = os.path.join(ROOT, "artifacts_local", ex.id, "ablate2048_f16")
    e.save_peft(d)
    g = e.to_gguf(d, d + ".gguf")
    rows = [json.loads(l) for l in open(os.path.join(DATA, "mbpp_full_train.jsonl"), encoding="utf-8")][:N]
    users = [f"{r['text']}\nYour code should pass this test:\n{r['test_list'][0]}" for r in rows]
    from sne.model import load
    model, tok = load(HF, ple_on_cpu=True)
    apply_torch(model, e, +1.0)
    prompts = render(tok, users)
    ref = []
    for i in range(0, N, 8):
        enc = tok(["<bos>" + p for p in prompts[i:i + 8]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("xpu")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=NEW, do_sample=False)
        ref += [tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True) for j in range(o.shape[0])]
    del model
    import gc
    gc.collect()
    torch.xpu.empty_cache()
    with LlamaServer(GGUF["f16"], loras=[g], parallel=8, log_path=os.path.join(ex.dir, "server.log")) as srv:
        on, _ = srv.complete_many(prompts, max_tokens=NEW, lora_scales={0: 1.0})
        off, _ = srv.complete_many(prompts, max_tokens=NEW, lora_scales={0: 0.0})
    with LlamaServer(GGUF["f16"], parallel=8, port=8093, log_path=os.path.join(ex.dir, "server_plain.log")) as srv:
        plain, _ = srv.complete_many(prompts, max_tokens=NEW)

    def agree(a, b):
        ta = [tok(x, add_special_tokens=False)["input_ids"] for x in a]
        tb = [tok(x, add_special_tokens=False)["input_ids"] for x in b]
        exact = sum(x == y for x, y in zip(ta, tb))
        fd = [next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y))) for x, y in zip(ta, tb)]
        return dict(exact=exact / N, mean_first_div=sum(fd) / N)
    out = dict(lora1_vs_torch_edit=agree([x["text"] for x in on], ref),
               lora0_vs_plain=agree([x["text"] for x in off], [x["text"] for x in plain]),
               lora1_vs_plain=agree([x["text"] for x in on], [x["text"] for x in plain]))
    ex.save_json("generations.json", dict(torch_edit=ref, lora1=[x["text"] for x in on], lora0=[x["text"] for x in off],
                                          plain=[x["text"] for x in plain]))
    ex.result(**out, headline=f"lora@1 vs torch edit exact {out['lora1_vs_torch_edit']['exact']:.2f}; "
                              f"lora@0 vs plain exact {out['lora0_vs_plain']['exact']:.2f}; edit changes "
                              f"{1-out['lora1_vs_plain']['exact']:.2f} of outputs")

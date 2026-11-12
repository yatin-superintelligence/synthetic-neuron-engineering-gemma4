"""EXP-094b: EXP-094's BOS value edit at F16 on the laptop (llama.cpp), the bf16 counterpart of EXP-094's W4A16 test on Colab.

Same edit and control as EXP-094 (D-066): at layers 0 to 14, W_v' = W_v - (W_v x^)(x^)^T. Here x^ is the unit vector of BOS's normed
v_proj input; this zeroes BOS's attention value in all 35 attention calls. The control uses the same rank-1 projection along a
random unit direction (seed 94). MBPP+ all 378 problems at F16, greedy, up to 8,192 tokens, unedited against bos against control.
Exact McNemar and paired bootstrap per suite (sne.runner's paired comparisons). The vectors come from a separate process, so the HF
model and llama-server never share the laptop's memory.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import MODELS, ROOT, SMOKE  # noqa: E402

OUT = os.path.join(ROOT, "scratch", "preflight", "EXP-094b") if SMOKE else os.path.join(ROOT, "artifacts_local", "EXP-094b")
VEC = os.path.join(OUT, "vectors.pt")


def phase1():
    from sne.model import layers, load
    os.makedirs(OUT, exist_ok=True)
    model, tok = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    Ls = layers(model)
    st = {}
    hooks = [Ls[i].self_attn.v_proj.register_forward_pre_hook(lambda m, a, i=i: st.__setitem__(i, a[0][0, 0].float().cpu())) for i in range(15)]
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(input_ids=torch.tensor([[tok.bos_token_id]], device=dev), use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()
    g = torch.Generator().manual_seed(94)
    vec = {}
    for i in range(15):
        xh = st[i] / st[i].norm()
        W = Ls[i].self_attn.v_proj.weight.float().cpu()
        r = torch.randn(xh.shape, generator=g)
        rh = r / r.norm()
        vec[i] = dict(A=xh, B=-(W @ xh), Ac=rh, Bc=-(W @ rh))
    torch.save(vec, VEC)
    print("[bos] vectors saved for 15 layers", flush=True)


if __name__ == "__main__" and "--phase1" in sys.argv:
    phase1()
    sys.exit(0)

from sne.artifacts import build_adapters  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402


def edit(vec, ctrl):
    def fn(e, bw, prec):
        for i, v in vec.items():
            A, B = (v["Ac"], v["Bc"]) if ctrl else (v["A"], v["B"])
            e.add(i, "self_attn.v_proj", B.view(-1, 1), A.view(1, -1))
        e.meta.update(layers=list(vec), rank=1, kind="control" if ctrl else "bos value deletion")
    return fn


with Experiment("EXP-094b", "bos-value-edit-f16", category="improvement", depends_on=["EXP-094"],
                motivation="bf16 counterpart of EXP-094 (W4A16 on Colab): the same exact rank-1 BOS value edit and control at F16 on llama.cpp.",
                hypothesis="As D-066: the BOS value edit raises MBPP+ passes against the unedited model and the control (exact McNemar p < 0.05).",
                config=dict(layers=list(range(15)), suites=["mbpp_dev", "mbpp_test"], precision="f16", control="random unit direction, seed 94")) as ex:
    os.makedirs(OUT, exist_ok=True)
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--phase1"], check=True)
    vec = torch.load(VEC)
    ad = {n: build_adapters(ex, n, edit(vec, n == "control"), precisions=("f16",)) for n in ("bos", "control")}
    arms = [Arm("base_f16", "f16"), Arm("bos_f16", "f16", lora=ad["bos"]["f16"]), Arm("control_f16", "f16", lora=ad["control"]["f16"])]
    res, comps = evaluate_arms(ex, arms, ["mbpp_dev", "mbpp_test"], reference="base", max_tokens=8192)
    tab = {a.name: {s: res[a.name][s]["passed"] for s in ("mbpp_dev", "mbpp_test")} for a in arms}
    ex.result(table=tab, paired=comps, headline=f"F16 MBPP+ dev+test: unedited {sum(tab['base_f16'].values())}, bos {sum(tab['bos_f16'].values())}, "
                                               f"control {sum(tab['control_f16'].values())} of 378")

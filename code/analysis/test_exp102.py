import os, sys, importlib.util, torch, json
os.environ["EXP102_UNIT"] = "1"
spec = importlib.util.spec_from_file_location("x102", "experiments/scripts/exp102_neuron_transplant.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
M.K = 8
torch.manual_seed(0)
d2, dd, V = 16, 24, 50
nst, per = 20, 30
data_ids = []
for k in range(nst):
    pos = list(range(per)); ans_idx = list(range(20, 30))
    data_ids.append(dict(pos=pos, ans_pos=[100 + i for i in ans_idx], ans_idx=ans_idx, gold=torch.randint(0, V, (10,)).tolist()))
N = nst * per
def info(d, nffs, lay):
    W = {L: dict(G=torch.randn(nffs(L), d), U=torch.randn(nffs(L), d), D=torch.randn(d, nffs(L)), wp=torch.rand(d) + 0.1, s=0.9) for L in lay}
    return dict(W=W, s=[0.9] * 35, wf=torch.rand(d) + 0.5, E=torch.randn(V, d), n=35)
LD = {L: L + 2 for L in M.L2}
I2 = info(d2, lambda L: 32 if L < 15 else 64, M.L2)
Id = info(dd, lambda L: 40, [LD[L] for L in M.L2])
Id["s"] = [0.9] * 42
A = torch.randn(N, d2)
rec2 = {L: dict(x=A + 0.1 * torch.randn(N, d2), h=A @ torch.randn(d2, d2)) for L in M.L2}
recd = {LD[L]: dict(x=A @ torch.randn(d2, dd), h=A @ torch.randn(d2, dd)) for L in M.L2}
C = dict(I2=I2, Id=Id, rec2=rec2, recd=recd, t2=torch.randint(0, V, (nst * 10,)), td=torch.randint(0, V, (nst * 10,)), rms2=torch.rand(nst * 10) + 1, rmsd=torch.rand(nst * 10) + 1, LD=LD, same_tok=True)
rep, fits = M.analyse(C, data_ids, print)
arms = M.write_arms(fits, I2, Id, LD, "scratch/analysis/x102_test")
print(sorted(arms))
from safetensors.torch import load_file
for a, p in arms.items():
    sd = load_file(p + "/adapter_model.safetensors"); cfg = json.load(open(p + "/adapter_config.json"))
    # verify: applying the LoRA reproduces the intended replacement
    L = cfg["layers_to_transform"][-1]
    base = f"base_model.model.model.language_model.layers.{L}.mlp"
    D = I2["W"][L]["D"] + (sd[base + ".down_proj.lora_B.weight"].float() @ sd[base + ".down_proj.lora_A.weight"].float())
    J = fits[L]["J"]
    if a == "abl":
        print(a, "layer", L, "replaced cols now zero:", float(D[:, J].abs().max()) < 1e-2, "others unchanged:", float((D - I2["W"][L]["D"]).abs().sum() - I2["W"][L]["D"][:, J].abs().sum()) < 1e-1)
    else:
        G = I2["W"][L]["G"] + sd[base + ".gate_proj.lora_B.weight"].float() @ sd[base + ".gate_proj.lora_A.weight"].float()
        which = [w for n, w, al in M.ARMS if n == a][0]; idx, Vm = fits[L]["sel"][which]
        Gt = Id["W"][LD[L]]["G"][idx] @ fits[L]["P"].T
        print(a, "layer", L, "rank", cfg["r"], cfg["target_modules"], "gate rows match:", torch.allclose(G[J[:len(idx)]], Gt, atol=0.05 * float(Gt.abs().max())), "n", len(idx))
print(json.dumps(rep["layers"][32])[:600])

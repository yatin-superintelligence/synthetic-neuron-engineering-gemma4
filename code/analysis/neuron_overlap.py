"""D-228: are the neurons that mark success (EXP-154, correlation) the neurons that cause the passing fix (EXP-160, EXP-162b,
attribution)? And is the causal set stable between the two causal searches? Chance overlap of sets A and B among the 1,290,240
MLP neurons of 31B is |A| x |B| / 1,290,240.
Usage: python tools/analysis/neuron_overlap.py"""
import collections
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
N = 60 * 21504
sets = {}
for name, path, k in (("markers (EXP-154)", "artifacts_local/EXP-154/neurons.pt", 64), ("causal (EXP-160)", "artifacts_local/EXP-160/neurons.pt", None),
                      ("causal (EXP-162b)", "artifacts_local/EXP-162b/neurons.pt", None)):
    top = torch.load(path, weights_only=False)["top"]
    top = top[:k] if k else top
    sets[name] = {(t["layer"], t["neuron"]): t.get("sign") for t in top}
names = list(sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = sets[names[i]], sets[names[j]]
        shared = set(a) & set(b)
        same = sum(1 for q in shared if a[q] is not None and a[q] == b[q])
        print(f"{names[i]} ({len(a)}) & {names[j]} ({len(b)}): {len(shared)} shared (same sign {same}); chance {len(a) * len(b) / N:.4f}")
for name, s in sets.items():
    c = collections.Counter(l for l, _ in s)
    print(f"{name}: layers 0-23 {sum(v for l, v in c.items() if l <= 23)}, 24-55 {sum(v for l, v in c.items() if 24 <= l <= 55)}, "
          f"56-59 {sum(v for l, v in c.items() if l >= 56)}")

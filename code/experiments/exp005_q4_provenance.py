"""EXP-005: what exactly is the Q4_0 arm? Compare Google's QAT Q4_0 GGUF with our own Q4_0 quantisation of the QAT
full-precision checkpoint (llama-quantize b12129), tensor by tensor.

If they are bit-identical (or nearly), the Q4_0 arm is 'the QAT checkpoint at 4 bits' and edits derived from the
full-precision QAT weights are on the right base. If not, we report how they differ (types per tensor, relative
error of each to the F16 weights).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402

from sne import MODELS, ROOT  # noqa: E402
from sne.registry import Experiment  # noqa: E402

GG = os.path.join(MODELS, "gguf")
GOOGLE = os.path.join(GG, "gemma-4-E2B_q4_0-it.gguf")
F16 = os.path.join(GG, "e2b-qat-f16.gguf")
OURS = os.path.join(ROOT, "artifacts_local", "EXP-005", "e2b-qat-ours-q4_0.gguf")

with Experiment("EXP-005", "q4-provenance", category="infrastructure", depends_on=["EXP-001"],
                motivation="Know precisely what the Q4_0 arm is before attributing anything to quantisation.",
                hypothesis="Block-quantised linear weights are identical or nearly identical; embeddings/norms may use "
                           "different tensor types in Google's file.",
                config=dict(google=GOOGLE, f16=F16, ours=OURS, quantizer="llama-quantize Q4_0")) as ex:
    import gguf
    os.makedirs(os.path.dirname(OURS), exist_ok=True)
    if not os.path.exists(OURS):
        p = subprocess.run([os.path.join(ROOT, "tools", "llama", "vulkan", "llama-quantize.exe"), F16, OURS, "Q4_0"],
                           capture_output=True, text=True)
        ex.log(p.stdout[-1200:])
    rg, ro, rf = gguf.GGUFReader(GOOGLE), gguf.GGUFReader(OURS), gguf.GGUFReader(F16)
    tg, to, tf = ({t.name: t for t in r.tensors} for r in (rg, ro, rf))
    rows, same_type, identical = [], 0, 0
    for name in sorted(tg):
        a, b = tg[name], to.get(name)
        if b is None:
            rows.append(dict(name=name, note="missing in ours"))
            continue
        st = a.tensor_type == b.tensor_type
        same_type += st
        ident = st and np.array_equal(np.asarray(a.data), np.asarray(b.data))
        identical += ident
        row = dict(name=name, google=a.tensor_type.name, ours=b.tensor_type.name, identical=bool(ident))
        if name.startswith("blk.") and (name.endswith("ffn_down.weight") or name.endswith("attn_q.weight")) and \
                int(name.split(".")[1]) % 7 == 0:
            ref = gguf.quants.dequantize(tf[name].data, tf[name].tensor_type).astype(np.float32)
            ga = gguf.quants.dequantize(a.data, a.tensor_type).astype(np.float32)
            ob = gguf.quants.dequantize(b.data, b.tensor_type).astype(np.float32)
            row.update(rel_err_google=float(np.linalg.norm(ga - ref) / np.linalg.norm(ref)),
                       rel_err_ours=float(np.linalg.norm(ob - ref) / np.linalg.norm(ref)))
        rows.append(row)
    ex.save_json("tensor_comparison.json", rows)
    types = {}
    for r in rows:
        if "google" in r:
            types[f"{r['google']}->{r['ours']}"] = types.get(f"{r['google']}->{r['ours']}", 0) + 1
    errs = [r for r in rows if "rel_err_google" in r]
    ex.result(n_tensors=len(tg), same_type=same_type, bit_identical=identical, type_pairs=types, sampled_errors=errs,
              headline=f"{identical}/{len(tg)} tensors bit-identical, {same_type} same type; type pairs {types}")
    os.remove(OURS)

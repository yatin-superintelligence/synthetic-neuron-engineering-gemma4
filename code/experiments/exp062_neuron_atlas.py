"""EXP-062: neuron atlas of real coding work. Which neurons respond to what, measured on real agent states.

All 338,000 or so MLP neurons in all 35 layers, across real agent conversations from the official runs, dev split, full response
plus the last 3000 tokens of context, and the first 3000 tokens once per task, teacher-forced at bf16. Each token is labelled by what the model is doing (issue, thinking,
tool syntax, old_string, new_string, command, path, tool response) and, inside code, by lexical class. For each neuron and class:
Cohen's d against all other tokens. Tasks are split in two, so each neuron's selectivity is measured twice on separate tasks. A
class counts only if selectivity replicates between halves, and only neurons at or above d = 1 in both halves are reported.
Neuron groups come from clustering replicated neurons by their response profile. Writes are read through the token embeddings only
in the last layers, where that reading means something. Everything is measurement, with no selection of what to find. This shows association. EXP-063 tests cause by
switching groups off. Outputs hold numbers and vocabulary tokens only, with no task text.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
from scipy.cluster.vq import kmeans2  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from sne import MODELS, ROOT, SMOKE, atlas as A, editbench as E, loops as L  # noqa: E402
from sne.conditional import call_text  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import tokenizer  # noqa: E402

CTX, D_MIN, K = 3000, 1.0, 16
OUT = os.path.join(ROOT, "artifacts_local", "EXP-062")

with Experiment("EXP-062", "neuron-atlas-real-coding", category="localization", depends_on=["EXP-041"],
                motivation="The field needs measured maps, not guesses. No one has mapped which Gemma neurons respond to which part of "
                           "agentic coding, and every neuron edit should start from such a map.",
                hypothesis="No directional claim: this is a measurement. Reported as found, including if nothing replicates.",
                config=dict(ctx_tokens=CTX, d_min=D_MIN, clusters=K, classes=A.CLASSES, split="dev", half_seed=62)) as ex:
    tok = tokenizer()
    cases = E.load("dev", per_run_task=4)[:4 if SMOKE else None]
    tasks = sorted({c["instance_id"] for c in cases})
    random.Random(62).shuffle(tasks)
    half_a = set(tasks[: len(tasks) // 2])
    ex.log("cases", len(cases), "tasks", len(tasks))
    from sne.model import load
    model, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    Ls = model.model.language_model.layers
    widths = [l.mlp.down_proj.weight.shape[1] for l in Ls]
    acc = {"a": A.Accumulator(widths), "b": A.Accumulator(widths)}
    idx = {c: i for i, c in enumerate(A.CLASSES)}
    t0 = time.time()
    head_done = set()
    for k, c in enumerate(cases):
        text = L.render(tok, c) + c["thought"] + "<channel|>" + call_text(c["args"])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        seg, lex = A.label_tokens(tok.batch_decode([[i] for i in ids]))
        n_resp = len(tok(c["thought"] + "<channel|>" + call_text(c["args"]), add_special_tokens=False)["input_ids"])
        keep = len(ids) - min(len(ids), CTX + n_resp)
        windows = [(keep, len(ids))]
        if c["instance_id"] not in head_done and keep > 0:  # one head window per task, so system prompt and issue are measured too
            windows.append((0, min(len(ids), CTX)))
            head_done.add(c["instance_id"])
        for lo, hi in windows:
            M = np.zeros((hi - lo + 1, len(A.CLASSES)), dtype=np.float32)
            for t, (s, x) in enumerate(zip(seg[lo:hi], lex[lo:hi]), 1):
                M[t, idx[s]] = 1
                if x:
                    M[t, idx[x]] = 1
            A.collect(model, [tok.bos_token_id] + ids[lo:hi], M, acc["a" if c["instance_id"] in half_a else "b"])
        if k % 10 == 0:
            ex.log(k, f"{time.time() - t0:.0f}s")
    all_acc = acc["a"].merged(acc["b"])
    ex.log("tokens per class", {c: int(all_acc.n[idx[c]]) for c in A.CLASSES})
    rows, profile_a, profile_b = [], [], []
    offsets = np.cumsum([0] + widths)
    for c in A.CLASSES:
        i = idx[c]
        da, db = np.concatenate(acc["a"].cohen_d(i)), np.concatenate(acc["b"].cohen_d(i))
        profile_a.append(da)
        profile_b.append(db)
        rep = (da >= D_MIN) & (db >= D_MIN)
        by_layer = [int(rep[offsets[l]:offsets[l + 1]].sum()) for l in range(len(widths))]
        order = np.argsort(-np.minimum(da, db))[:20]
        rows.append(dict(cls=c, tokens=int(all_acc.n[i]), spearman_between_halves=round(float(spearmanr(da, db).correlation), 3), n_replicated=int(rep.sum()),
                         by_layer=by_layer, top=[[int(np.searchsorted(offsets, g, side="right") - 1), int(g - offsets[np.searchsorted(offsets, g, side="right") - 1]), round(float(da[g]), 2), round(float(db[g]), 2)] for g in order]))
        ex.log(rows[-1]["cls"], "tokens", rows[-1]["tokens"], "rho", rows[-1]["spearman_between_halves"], "replicated", rows[-1]["n_replicated"])
    ex.save_json("classes.json", rows)
    Pa, Pb = np.stack(profile_a).T, np.stack(profile_b).T
    P = np.minimum(np.abs(Pa), np.abs(Pb)) * np.sign(Pa) * (np.sign(Pa) == np.sign(Pb))
    chosen = np.where(np.abs(P).max(1) >= D_MIN)[0]
    ex.log("neurons with a replicated response", len(chosen), "of", P.shape[0])
    groups = []
    if len(chosen) >= K:
        centers, lab = kmeans2(P[chosen].astype(np.float64), K, minit="++", seed=62)
        for g in range(K):
            mem = chosen[lab == g]
            if not len(mem):
                continue
            layer = np.searchsorted(offsets, mem, side="right") - 1
            pairs = [(int(l), int(m - offsets[l])) for l, m in zip(layer, mem)]
            prof = centers[g]
            top = np.argsort(-np.abs(prof))[:4]
            groups.append(dict(group=g, size=len(mem), layers=np.bincount(layer, minlength=len(widths)).tolist(), profile={A.CLASSES[t]: round(float(prof[t]), 2) for t in top},
                               lens=A.logit_lens(model, tok, pairs[:4096]), neurons=pairs[:4096]))
            ex.log("group", g, "size", len(mem), groups[-1]["profile"], groups[-1]["lens"]["promotes"][:5])
    ex.save_json("groups.json", groups)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "profiles.npz"), Pa=Pa.astype(np.float32), Pb=Pb.astype(np.float32), offsets=offsets)
    ok = [r for r in rows if r["n_replicated"]]
    ex.result(classes=[{k: r[k] for k in ("cls", "tokens", "spearman_between_halves", "n_replicated")} for r in rows], groups=len(groups), responding=int(len(chosen)), total=int(P.shape[0]),
              headline=f"{len(chosen)} of {P.shape[0]} neurons have a replicated class response; largest classes " + ", ".join(f"{r['cls']} {r['n_replicated']}" for r in sorted(ok, key=lambda r: -r["n_replicated"])[:4]))

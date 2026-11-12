"""EXP-031: per-neuron activation statistics on the selection corpus (EXP-030), PyTorch bf16 reference model.

Corpora (response tokens only, teacher-forced):
  code_A / code_B   : the model's own greedy MBPP-train solutions, task halves A and B (seeded split)
  prose_A / prose_B : the model's own plain-English explanations of the same tasks
  pass / fail       : all code samples (greedy + 2 sampled) that pass / fail the original MBPP asserts
Outputs: stats_<corpus>.npz, neuron selectivity (Cohen's d, rms log-ratio, p(a>0) difference) for code-vs-prose
and pass-vs-fail, split-half reliability, per-layer distribution, ranked neuron lists for later experiments.
"""
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from sne import MODELS, ROOT  # noqa: E402
from sne.probes import collect, neuron_table, selectivity  # noqa: E402
from sne.artifacts import publish  # noqa: E402
from sne.registry import Experiment  # noqa: E402

SEED = 31
HF = os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized")
corpus_path = glob.glob(os.path.join(ROOT, "experiments", "EXP-030_*", "corpus.json"))[0]

with Experiment("EXP-031", "activation-stats-code-vs-prose", category="localization", depends_on=["EXP-030"],
                motivation="Locate MLP neurons whose activity is specific to writing code (vs. explaining the same "
                           "task in prose) and neurons that separate correct from incorrect code. Split-half "
                           "reliability tells us whether a 'code neuron' is a stable property or sampling noise.",
                hypothesis="A small fraction of neurons (<5%) are strongly code-selective (|d|>1), concentrated in "
                           "early-middle layers; selectivity is highly reliable across task halves (Spearman > 0.8). "
                           "Pass-vs-fail selectivity is much weaker and less reliable.",
                config=dict(seed=SEED, corpus=os.path.relpath(corpus_path, ROOT), max_len=1024, batch_size=4,
                            metrics=["cohen_d", "rms_log_ratio", "ppos_diff"])) as ex:
    corpus = json.load(open(corpus_path, encoding="utf-8"))
    idx = list(range(len(corpus)))
    random.Random(SEED).shuffle(idx)
    half = {i: ("A" if k < len(idx) // 2 else "B") for k, i in enumerate(idx)}
    sets = {"code_A": [], "code_B": [], "prose_A": [], "prose_B": [], "pass": [], "fail": []}
    for i, r in enumerate(corpus):
        sets[f"code_{half[i]}"].append((r["code_prompt"], r["code_greedy"]))
        sets[f"prose_{half[i]}"].append((r["prose_prompt"], r["prose"]))
        seen = set()
        for k in ["code_greedy", "code_s1", "code_s2"]:
            if r[k] in seen:
                continue
            seen.add(r[k])
            sets["pass" if r[f"pass_{k}"] else "fail"].append((r["code_prompt"], r[k]))
    ex.log({k: len(v) for k, v in sets.items()})
    from sne.model import load
    model, tok = load(HF)
    nL = len(model.model.language_model.layers)
    tabs, raw = {}, {}
    for name, pairs in sets.items():
        t = time.time()
        st = collect(model, tok, pairs, batch_size=4)
        raw[name] = st
        np.savez_compressed(os.path.join(ex.dir, f"stats_{name}.npz"), **st)
        tabs[name] = neuron_table(st, nL)
        ex.log(f"{name}: {len(pairs)} seqs, {float(st['n_0']):.0f} response tokens, {time.time()-t:.1f}s")
    # exact merge of the two halves: statistics are sums, so add them
    code = neuron_table({k: raw["code_A"][k] + raw["code_B"][k] for k in raw["code_A"]}, nL)
    prose = neuron_table({k: raw["prose_A"][k] + raw["prose_B"][k] for k in raw["prose_A"]}, nL)
    out = {"tokens": {k: tabs[k][0]["n"] for k in tabs}}
    for metric in ["cohen_d", "rms_log_ratio", "ppos_diff"]:
        dA = np.concatenate(selectivity(tabs["code_A"], tabs["prose_A"], metric))
        dB = np.concatenate(selectivity(tabs["code_B"], tabs["prose_B"], metric))
        pA = np.concatenate(selectivity(tabs["pass"], tabs["fail"], metric))
        rel = dict(spearman_code_vs_prose=float(spearmanr(dA, dB).correlation))
        for k in (256, 1024, 4096):
            ta, tb = set(np.argsort(-dA)[:k]), set(np.argsort(-dB)[:k])
            rel[f"jaccard_top{k}"] = len(ta & tb) / len(ta | tb)
        out[metric] = rel
        ex.log(metric, rel)
    # final scores and per-layer distribution (cohen_d on merged halves)
    d_code = selectivity(code, prose, "cohen_d")
    d_corr = selectivity(tabs["pass"], tabs["fail"], "cohen_d")
    sizes = [len(x) for x in d_code]
    offsets = np.cumsum([0] + sizes)
    flat = np.concatenate(d_code)
    flat_corr = np.concatenate(d_corr)
    per_layer = []
    for i in range(nL):
        di = d_code[i]
        per_layer.append(dict(layer=i, width=len(di), n_code_d_gt1=int((di > 1).sum()), n_prose_d_gt1=int((di < -1).sum()),
                              max_d=float(di.max()), min_d=float(di.min()),
                              n_corr_d_gt05=int((np.abs(d_corr[i]) > 0.5).sum())))
    ex.save_json("per_layer.json", per_layer)

    def to_list(order, k):
        res = []
        for g in order[:k]:
            L = int(np.searchsorted(offsets, g, side="right") - 1)
            res.append([L, int(g - offsets[L]), float(flat[g]), float(flat_corr[g])])
        return res
    rank = dict(code_top=to_list(np.argsort(-flat), 8192), prose_top=to_list(np.argsort(flat), 8192),
                pass_top=to_list(np.argsort(-flat_corr), 4096), fail_top=to_list(np.argsort(flat_corr), 4096),
                layer_widths=sizes)
    ex.save_json("neuron_rankings.json", rank)
    # neurons with lowest overall usage (candidates for recycling into synthetic neurons)
    usage = [np.maximum(code[i]["rms"], prose[i]["rms"]) for i in range(nL)]
    flat_u = np.concatenate(usage)
    order_u = np.argsort(flat_u)
    dead = []
    for g in order_u[:4096]:
        L = int(np.searchsorted(offsets, g, side="right") - 1)
        dead.append([L, int(g - offsets[L]), float(flat_u[g])])
    ex.save_json("low_usage_neurons.json", dead)
    # mean residual stream (after each layer) and mean MLP input per corpus: [corpus, layer, d_model]
    cs = ["code_A", "code_B", "prose_A", "prose_B", "pass", "fail"]
    np.save(os.path.join(ex.dir, "resid_means.npy"),
            np.stack([np.stack([raw[c][f"resid_sum_{i}"] / float(raw[c][f"n_{i}"]) for i in range(nL)]) for c in cs]))
    np.save(os.path.join(ex.dir, "mlpin_means.npy"),
            np.stack([np.stack([raw[c][f"mlpin_sum_{i}"] / float(raw[c][f"n_{i}"]) for i in range(nL)]) for c in cs]))
    tot = sum(sizes)
    out.update(total_neurons=tot, n_code_d_gt1=int((flat > 1).sum()), n_prose_d_gt1=int((flat < -1).sum()),
               n_corr_absd_gt05=int((np.abs(flat_corr) > 0.5).sum()),
               headline=f"{int((flat > 1).sum())}/{tot} neurons code-selective (d>1); split-half Spearman "
                        f"{out['cohen_d']['spearman_code_vs_prose']:.3f}; top-1024 Jaccard {out['cohen_d']['jaccard_top1024']:.3f}")
    # per-layer lowest-usage neurons (recycling candidates for synthetic / transplanted neurons)
    low_by_layer = {i: np.argsort(usage[i])[:512].tolist() for i in range(nL)}
    ex.save_json("low_usage_by_layer.json", low_by_layer)
    publish(ex, sorted(glob.glob(os.path.join(ex.dir, "*.npz"))) + sorted(glob.glob(os.path.join(ex.dir, "*.npy"))),
            f"{ex.id}_activation_stats.zip")
    ex.result(**out)

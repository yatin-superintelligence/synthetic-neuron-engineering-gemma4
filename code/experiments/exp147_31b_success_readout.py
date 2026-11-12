"""EXP-147 (D-202): can 31B's activations at its first real edit tell whether that run's fix will pass, across tasks?

Why: unedited 31B fails 13 of its 93 attempts at the 31 solvable tasks at the 8,192 setting, all with patches that fail the tests, and
on the 15 tasks it solves in 2 to 9 of 10 runs the knowledge is in the model. If the state at the first edit to a file the reference
fix touches separates runs that will pass from runs that will not, held out by task, there is a direction an edit can push toward;
if it does not, steering has nothing to push, and that is a result too.

Bench: artifacts_local/EXP-147/success_bench.jsonl.gz (tools/analysis/build_success_bench.py): one row per (31B run, task) with the
served request whose output makes that first edit, labelled by the verifier.
Measured, at 8 depths (decoder-layer outputs): the state at the last prompt token (before the edit is written) and the mean over the
edit's output tokens. Read-out: the task-centred mean difference between solved and failed rows, fitted on the other tasks and scored
on the held-out task (leave one task out); within-task pairwise AUC pooled over tasks (only pairs from the same task count, so task
difficulty cannot help). Null: labels shuffled within each task, the same selection over depths and positions (the maximum), 300
times. Gate (fixed before the run): pooled held-out AUC 0.70 or more with a permutation p below 0.05.
Also saved, for building and calibrating a steering unit on the laptop: at every depth, the least-used MLP neuron's three weight
vectors, the post-feed-forward norm weight and the layer scalar, and MLP inputs and outputs at 8 random positions per row.
Outputs: artifacts_local/EXP-147/{fit.json, capture.pt}. Runs on a Colab G4 (bf16 31B)."""
import gzip
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.model import layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("EXP140_MODEL", os.path.join(ROOT, "models", "gemma-4-E2B-it-qat-unquantized") if SMOKE else "google/gemma-4-31B-it-qat-q4_0-unquantized")
BENCH = os.path.join(ROOT, "artifacts_local", "EXP-147", "success_bench.jsonl.gz")
OUT = os.path.join(ROOT, "scratch", "preflight", "EXP-147") if SMOKE else os.path.join(ROOT, "artifacts_local", "EXP-147")
FRACTIONS = (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SAMPLES_PER_ROW, PERMS = (2, 20) if SMOKE else (8, 300)
rng = random.Random(147)


def auc_within(scores, solved, task):
    """Pairwise AUC over (solved, failed) pairs of the same task, pooled over tasks."""
    num = den = 0.0
    for t in set(task):
        ix = [i for i in range(len(task)) if task[i] == t]
        s = [scores[i] for i in ix if solved[i]]
        f = [scores[i] for i in ix if not solved[i]]
        for a in s:
            for b in f:
                num += 1.0 if a > b else 0.5 if a == b else 0.0
                den += 1
    return num / den if den else float("nan")


def loto_scores(X, solved, task):
    """Leave one task out: task-centred mean difference fitted on the other tasks, scored on the held-out one."""
    task = np.array(task)
    Xc = X.copy()
    for t in set(task.tolist()):
        m = task == t
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    sol = np.array(solved, bool)
    out = np.zeros(len(task))
    for t in set(task.tolist()):
        te, tr = task == t, task != t
        d = Xc[tr & sol].mean(0) - Xc[tr & ~sol].mean(0)
        out[te] = Xc[te] @ d
    return out


with Experiment("EXP-147", "31b-success-readout", category="diagnosis", depends_on=["EXP-139B", "EXP-144a", "EXP-145a"],
                motivation="31B's remaining failures on solvable tasks are wrong fixes on tasks it solves only sometimes; an edit can add solves "
                           "there only if the model's state at the fix already separates fixes that will pass from fixes that will not.",
                hypothesis="Held out by task, a task-centred mean-difference read-out at some depth ranks a run that will pass above one that will "
                           "not (same task) with pooled AUC 0.70 or more, permutation p < 0.05 after selection over depths and positions.",
                config=dict(model=REPO, bench="artifacts_local/EXP-147/success_bench.jsonl.gz", fractions=FRACTIONS, samples_per_row=SAMPLES_PER_ROW,
                            permutations=PERMS)) as ex:
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(BENCH, "rt", encoding="utf-8")]
    if SMOKE:
        two = [t for t in sorted({r['task'] for r in rows}) if 0 < sum(r['solved'] for r in rows if r['task'] == t) < sum(1 for r in rows if r['task'] == t)][:3]
        rows = [r for t in two for r in [x for x in rows if x['task'] == t and x['solved']][:2] + [x for x in rows if x['task'] == t and not x['solved']][:2]]
        for r in rows:
            r["prompt_ids"] = r["prompt_ids"][-600:]
            r["output_ids"] = r["output_ids"][:60]
    ex.log("bench rows", len(rows), "| solved", sum(r["solved"] for r in rows), "| tasks", len({r["task"] for r in rows}))
    from transformers import AutoModelForCausalLM
    from sne.chunked_attention import install
    install()
    path = REPO if os.path.isdir(REPO) else ("/content/hf_31b" if os.path.isdir("/content/hf_31b") else None)
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO, local_dir="/content/hf_31b")
    t0 = time.time()
    if torch.cuda.is_available():
        dev = "cuda"
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True).eval()
        for nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
            if hasattr(model.model, nm) and getattr(model.model, nm) is not None:
                setattr(model.model, nm, None)
    else:
        from sne.model import device as _device, load as _load
        model, _ = _load(path)
        dev = _device()
    Ls = layers(model)
    nL = len(Ls)
    LAY = sorted({min(nL - 1, int(round(f * nL)) - 1) for f in FRACTIONS})
    ex.log("model loaded in", round(time.time() - t0), "s | layers", nL, "| read at", LAY)

    st = {}
    hooks = []
    # the hooks keep only what is needed (the decision token, the edit's mean, the sampled positions, the neurons' summed activity),
    # never whole-sequence activations at 8 depths (about 14 GB at 24,000 tokens)
    def _out(m, a, o, l):
        h = (o[0] if isinstance(o, tuple) else o)[0]
        st[("last", l)] = h[st["P"] - 1].float().cpu()
        st[("outmean", l)] = h[st["P"]:].float().mean(0).cpu()

    def _mlp(m, a, o, l):
        st[("x", l)] = a[0][0][st["pos"]].half().cpu()
        st[("y", l)] = o[0][st["pos"]].half().cpu()

    def _down(m, a, l):
        st[("au", l)] = a[0][0].float().abs().sum(0)

    for l in LAY:
        hooks.append(Ls[l].register_forward_hook(lambda m, a, o, l=l: _out(m, a, o, l) or None))
        hooks.append(Ls[l].mlp.register_forward_hook(lambda m, a, o, l=l: _mlp(m, a, o, l) or None))
        hooks.append(Ls[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: _down(m, a, l)))
    feats = {(l, k): [] for l in LAY for k in ("last", "outmean")}
    samp = {(l, k): [] for l in LAY for k in ("x", "y")}
    usage = {l: None for l in LAY}
    keep = []
    t1 = time.time()
    try:
        for n, r in enumerate(rows):
            P = len(r["prompt_ids"])
            ids = r["prompt_ids"] + r["output_ids"]
            pos = sorted(rng.sample(range(len(ids)), min(SAMPLES_PER_ROW, len(ids))))
            st["P"], st["pos"] = P, torch.tensor(pos, device=dev)
            try:
                with torch.no_grad():
                    model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                ex.log("row", n, "skipped: out of memory at", len(ids), "tokens")
                continue
            for l in LAY:
                feats[(l, "last")].append(st[("last", l)])
                feats[(l, "outmean")].append(st[("outmean", l)])
                samp[(l, "x")].append(st[("x", l)])
                samp[(l, "y")].append(st[("y", l)])
                usage[l] = st[("au", l)] if usage[l] is None else usage[l] + st[("au", l)]
            keep.append(n)
            st.clear()
            if n % 10 == 0:
                ex.log(f"row {n + 1} of {len(rows)}, {len(ids)} tokens, {time.time() - t1:.0f} s so far")
    finally:
        for h in hooks:
            h.remove()
    rows = [rows[i] for i in keep]
    solved = [bool(r["solved"]) for r in rows]
    task = [r["task"] for r in rows]

    FX = {key: torch.stack(v).numpy().astype(np.float64) for key, v in feats.items()}
    res = {key: auc_within(loto_scores(X, solved, task).tolist(), solved, task) for key, X in FX.items()}
    best = max(res, key=lambda key: res[key])
    null = []
    for p in range(PERMS):
        perm = list(solved)
        for t in set(task):
            ix = [i for i in range(len(task)) if task[i] == t]
            vals = [perm[i] for i in ix]
            rng.shuffle(vals)
            for i, v in zip(ix, vals):
                perm[i] = v
        null.append(max(auc_within(loto_scores(X, perm, task).tolist(), perm, task) for X in FX.values()))
    pval = (1 + sum(x >= res[best] for x in null)) / (1 + len(null))
    gate = bool(res[best] >= 0.70 and pval < 0.05)
    per_task = {}
    for t in sorted(set(task)):
        per_task[t] = f"{sum(s for s, tt in zip(solved, task) if tt == t)} of {task.count(t)}"
    out = dict(rows=len(rows), solved=sum(solved), tasks=per_task, layers=LAY,
               auc={f"L{l} {k}": round(v, 3) for (l, k), v in sorted(res.items())}, best=dict(layer=best[0], position=best[1], auc=round(res[best], 3)),
               permutation=dict(n=PERMS, null_max_auc_median=round(float(np.median(null)), 3), null_max_auc_95=round(float(np.percentile(null, 95)), 3), p=round(pval, 4)),
               gate_passed=gate)
    ex.log("AUC by depth and position:", out["auc"])
    ex.log("best", out["best"], "| permutation", out["permutation"], "| gate", gate)

    units = {}
    for l in LAY:
        mlp = Ls[l].mlp
        i = int(torch.argmin(usage[l]).item())
        units[l] = dict(i=i, w_gate=mlp.gate_proj.weight[i].float().cpu(), w_up=mlp.up_proj.weight[i].float().cpu(), w_down=mlp.down_proj.weight[:, i].float().cpu(),
                        w_post=Ls[l].post_feedforward_layernorm.weight.float().cpu(), eps=float(Ls[l].post_feedforward_layernorm.eps),
                        layer_scalar=float(getattr(Ls[l], "layer_scalar", torch.ones(1)).float().item()), n_inter=int(mlp.down_proj.in_features))
    torch.save(dict(rows=[{k: r[k] for k in ("run", "iid", "task", "step", "solved", "file")} for r in rows], layers=LAY,
                    feats={f"{l}|{k}": torch.stack(v) for (l, k), v in feats.items()},
                    samples={f"{l}|{k}": torch.cat(v) for (l, k), v in samp.items()}, units=units), os.path.join(OUT, "capture.pt"))
    ex.save_json("fit.json", out)
    ex.result(headline=f"held-out within-task AUC {out['best']['auc']} at layer {best[0]} ({best[1]}), permutation p {out['permutation']['p']}; gate {'passed' if gate else 'not passed'}",
              fit=out)

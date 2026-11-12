"""EXP-154 (D-213): which of 31B's 1,290,240 MLP neurons (60 layers x 21,504) separate runs that will solve a task from runs that
will not, at the first edit to a file the reference fix touches, held out by task?

Bench: EXP-147's (artifacts_local/EXP-147/success_bench.jsonl.gz): 123 first edits by 31B in its own logged runs of the 15 tasks it
solves only sometimes, labelled by the verifier. Measured per neuron (its activation entering down_proj): at the last prompt token
(before the edit is written) and as the mean over the edit's output tokens: 2,580,480 features.
Score of a feature: within-task pairwise AUC (only a solved and a failed run of the same task are compared), pooled over tasks.
Held out by task, two folds (tasks split by a hash): the K = 64 features furthest from 0.5 on one half are combined (sign-aligned,
task-centred, scaled by their spread on that half) and the combined score's AUC is measured on the other half; the two folds' AUCs
are averaged. Null: labels shuffled within each task and the whole selection repeated, 200 times. Gate (fixed before the run):
held-out AUC 0.70 or more and permutation p below 0.05. Saved for an edit either way: the 64 features chosen on all tasks, with their
layers, indices, signs, full-data AUCs, activation means for solved and failed runs, and down_proj columns.
Outputs: artifacts_local/EXP-154/{fit.json, neurons.pt}. Runs on a Colab G4 (bf16 31B)."""
import gzip
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.model import layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("SCAN_MODEL") or os.environ.get("EXP140_MODEL") or (os.path.join(ROOT, "models", "gemma-4-E2B-it-qat-unquantized") if SMOKE else "google/gemma-4-31B-it-qat-q4_0-unquantized")
SCAN_ID = os.environ.get("SCAN_ID", "EXP-154")  # the neuron scan as a tool: any model, any bench (EXP-158: 12B)
BENCH = os.environ.get("SCAN_BENCH") or os.path.join(ROOT, "artifacts_local", "EXP-147", "success_bench.jsonl.gz")
OUT = os.path.join(ROOT, "scratch", "preflight", SCAN_ID) if SMOKE else os.path.join(ROOT, "artifacts_local", SCAN_ID)
K, PERMS = (8, 10) if SMOKE else (64, 200)


def fold(task):
    return zlib.crc32(task.encode()) % 2


def auc_features(X, solved, task, tasks):
    """Pooled within-task AUC of every column of X (rows x features), over the given tasks."""
    num = torch.zeros(X.shape[1], device=X.device)
    den = 0
    for t in tasks:
        ix = [i for i in range(len(task)) if task[i] == t]
        s = [i for i in ix if solved[i]]
        f = [i for i in ix if not solved[i]]
        if not s or not f:
            continue
        S, Fm = X[s].float(), X[f].float()
        for a in range(len(s)):  # rows one at a time: (failed rows x features) stays small
            d = S[a][None, :] - Fm
            num += (d > 0).float().sum(0) + 0.5 * (d == 0).float().sum(0)
        den += len(s) * len(f)
    return num / max(den, 1), den


def combined_auc(X, solved, task, sel_tasks, test_tasks, k):
    a, _ = auc_features(X, solved, task, sel_tasks)
    idx = torch.topk((a - 0.5).abs(), k).indices
    sign = torch.sign(a[idx] - 0.5)
    Z = X[:, idx].float()
    rows_sel = [i for i in range(len(task)) if task[i] in sel_tasks]
    Zc = Z.clone()
    for t in set(task):  # task-centred
        m = [i for i in range(len(task)) if task[i] == t]
        Zc[m] -= Zc[m].mean(0, keepdim=True)
    sd = Zc[rows_sel].std(0) + 1e-6
    score = (Zc / sd * sign).mean(1)
    b, den = auc_features(score[:, None], solved, task, test_tasks)
    return float(b[0]), den, idx, sign, a


with Experiment(SCAN_ID, "31b-success-neurons" if SCAN_ID == "EXP-154" else os.environ.get("SCAN_SLUG", "success-neurons"), category="diagnosis", depends_on=["EXP-147"],
                motivation="The author asked which neurons contribute to solving tasks. EXP-147 read the residual stream at 8 depths; this "
                           "scans every MLP neuron of 31B at the fix, with selection and test on different tasks.",
                hypothesis="64 neurons chosen on one half of the tasks rank a passing run above a failing run of the same task on the other "
                           "half with AUC 0.70 or more, permutation p < 0.05 with the selection repeated under shuffled labels.",
                config=dict(model=REPO, bench=os.path.relpath(BENCH, ROOT), k=K, permutations=PERMS)) as ex:
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(BENCH, "rt", encoding="utf-8")]
    if SMOKE:
        two = [t for t in sorted({r["task"] for r in rows}) if 0 < sum(r["solved"] for r in rows if r["task"] == t) < sum(1 for r in rows if r["task"] == t)][:4]
        rows = [r for t in two for r in [x for x in rows if x["task"] == t and x["solved"]][:2] + [x for x in rows if x["task"] == t and not x["solved"]][:2]]
        for r in rows:
            r["prompt_ids"] = r["prompt_ids"][-500:]
            r["output_ids"] = r["output_ids"][:40]
    ex.log("bench rows", len(rows), "| solved", sum(r["solved"] for r in rows), "| tasks", len({r["task"] for r in rows}))
    from transformers import AutoModelForCausalLM
    from sne.chunked_attention import install
    install()
    cache = "/content/hf_" + os.path.basename(REPO.rstrip("/")).lower()
    path = REPO if os.path.isdir(REPO) else (cache if os.path.isdir(cache) else None)
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO, local_dir=cache)
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
    ex.log("model loaded in", round(time.time() - t0), "s | layers", nL)
    st = {}

    def _down(m, a, l):
        h = a[0][0]
        st[("last", l)] = h[st["P"] - 1].half().cpu()
        st[("outmean", l)] = h[st["P"]:].float().mean(0).half().cpu()

    hooks = [Ls[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: _down(m, a, l)) for l in range(nL)]
    feats, keep = [], []
    t1 = time.time()
    try:
        for n, r in enumerate(rows):
            st["P"] = len(r["prompt_ids"])
            ids = r["prompt_ids"] + r["output_ids"]
            try:
                with torch.no_grad():
                    model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                ex.log("row", n, "skipped: out of memory at", len(ids), "tokens")
                continue
            feats.append(torch.cat([st[(k, l)] for k in ("last", "outmean") for l in range(nL)]))
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
    n_inter = max(Ls[l2].mlp.down_proj.in_features for l2 in range(nL))
    adev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.stack(feats).to(adev)  # rows x (2 kinds x layers x neurons), fp16
    ex.log("features", tuple(X.shape))
    both = sorted(t for t in set(task) if 0 < sum(s for s, tt in zip(solved, task) if tt == t) < task.count(t))
    order = sorted(both, key=lambda t: zlib.crc32(t.encode()))  # a balanced split: tasks ordered by a hash, alternately to A and B
    A, B = order[0::2], order[1::2]
    ex.log("tasks with both outcomes", len(both), "| fold A", A, "| fold B", B)
    k = min(K, X.shape[1])
    ab, dab, _, _, _ = combined_auc(X, solved, task, A, B, k)
    ba, dba, _, _, _ = combined_auc(X, solved, task, B, A, k)
    obs = (ab * dab + ba * dba) / max(dab + dba, 1)
    ex.log("held-out AUC: select on A test on B", round(ab, 3), f"({dab} pairs) | select on B test on A", round(ba, 3), f"({dba} pairs) | pooled", round(obs, 3))
    rng = np.random.default_rng(154)
    null = []
    for p in range(PERMS):
        perm = list(solved)
        for t in set(task):
            ix = [i for i in range(len(task)) if task[i] == t]
            vals = [perm[i] for i in ix]
            rng.shuffle(vals)
            for i, v in zip(ix, vals):
                perm[i] = v
        a1, d1, _, _, _ = combined_auc(X, perm, task, A, B, k)
        a2, d2, _, _, _ = combined_auc(X, perm, task, B, A, k)
        null.append((a1 * d1 + a2 * d2) / max(d1 + d2, 1))
        if p % 50 == 0:
            ex.log(f"permutation {p + 1} of {PERMS}")
    pval = (1 + sum(x >= obs for x in null)) / (1 + len(null))
    gate = bool(obs >= 0.70 and pval < 0.05)
    full, _ = auc_features(X, solved, task, both)
    idx = torch.topk((full - 0.5).abs(), k).indices.cpu()
    top = []
    sol_rows = [i for i in range(len(rows)) if solved[i]]
    fail_rows = [i for i in range(len(rows)) if not solved[i]]
    sizes = [Ls[l2].mlp.down_proj.in_features for l2 in range(nL)]  # layers may differ in width
    starts, acc = [], 0
    for kind2 in ("last", "outmean"):
        for l2 in range(nL):
            starts.append((acc, kind2, l2))
            acc += sizes[l2]
    for j in idx.tolist():
        st0, kind, l = max((x for x in starts if x[0] <= j), key=lambda x: x[0])
        i = j - st0
        top.append(dict(feature=j, kind=kind, layer=int(l), neuron=int(i), auc=round(float(full[j]), 4),
                        mean_solved=round(float(X[sol_rows, j].float().mean()), 5), mean_failed=round(float(X[fail_rows, j].float().mean()), 5)))
    layer_counts = {}
    for t in top:
        layer_counts[t["layer"]] = layer_counts.get(t["layer"], 0) + 1
    torch.save(dict(top=top, w_down={f"{t['layer']}|{t['neuron']}": Ls[t["layer"]].mlp.down_proj.weight[:, t["neuron"]].float().cpu() for t in top},
                    n_inter=n_inter, layers=nL, rows=[{k2: r[k2] for k2 in ("run", "iid", "task", "solved")} for r in rows],
                    top_values=X[:, idx.to(adev)].float().cpu()), os.path.join(OUT, "neurons.pt"))
    out = dict(rows=len(rows), solved=sum(solved), tasks_with_both=len(both), features=int(X.shape[1]), k=k,
               held_out=dict(a_to_b=round(ab, 4), b_to_a=round(ba, 4), pooled=round(obs, 4), pairs=dab + dba),
               permutation=dict(n=PERMS, null_median=round(float(np.median(null)), 4), null_95=round(float(np.percentile(null, 95)), 4), p=round(pval, 4)),
               gate_passed=gate, top10=top[:10], top_layers=dict(sorted(layer_counts.items())))
    ex.log("held-out", out["held_out"], "| permutation", out["permutation"], "| gate", gate)
    ex.log("top neurons", top[:10])
    ex.save_json("fit.json", out)
    ex.result(headline=f"64 neurons chosen on half the tasks: held-out AUC {out['held_out']['pooled']} (null median {out['permutation']['null_median']}, p {out['permutation']['p']}); gate {'passed' if gate else 'not passed'}",
              fit=out)

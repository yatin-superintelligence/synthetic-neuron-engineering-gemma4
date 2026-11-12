"""EXP-160 (D-219): which of 31B's MLP neurons CAUSE it to prefer the edit that solves a task? An interventional search, after
EXP-154/155/156/157 showed that neurons chosen by correlation with success (they mark it) are not levers (editing them never raised
tasks solved).

States: EXP-147's bench (first edits to a reference-fix file). For every failing run of a task that also has a passing run, the
context is the failing run's prompt plus its own thought up to its tool call; the two continuations are its own edit call (a) and a
passing run's edit call for the same task (b). Preference = mean log-probability of b minus that of a (per token).
Attribution (one backward pass per continuation, through the call tokens only; the prompt is cached without gradients): for every
neuron entering down_proj, sum over the call's positions of activation x d(mean log p)/d(activation), b minus a. Positive: turning
the neuron up moves 31B toward the passing call.
Selection and causal test, held out by task (two folds, balanced hash split as EXP-154): the K neurons with the largest positive and
the K with the largest negative mean attribution on one half; on the other half the neurons are actually scaled (outputs x(1+alpha)
for positive, x(1-alpha) for negative, at every position) and the preference recomputed exactly, against K + K random neurons
scaled the same way. Gate (fixed before the run): in pooled held-out pairs the chosen neurons raise the preference in at least 65%
of pairs and their mean change exceeds the random neurons'. Saved: the neurons chosen on all pairs with their down_proj columns.
Outputs: artifacts_local/EXP-160/{fit.json, neurons.pt}. Runs on a Colab G4 (bf16 31B)."""
import gzip
import json
import os
import random
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.model import layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("SCAN_MODEL") or (os.path.join(ROOT, "models", "gemma-4-E2B-it-qat-unquantized") if SMOKE else "google/gemma-4-31B-it-qat-q4_0-unquantized")
BENCH = os.environ.get("SCAN_BENCH") or os.path.join(ROOT, "artifacts_local", "EXP-147", "success_bench.jsonl.gz")
OUT = os.path.join(ROOT, "scratch", "preflight", "EXP-160") if SMOKE else os.path.join(ROOT, "artifacts_local", "EXP-160")
TC0 = 48
K, ALPHA, MAX_CALL, PAIRS_PER_FAIL = (4, 1.0, 40, 1) if SMOKE else (32, 1.0, 64, 1)  # 64 call tokens: backward through attention over a 10,000-token cache ran out of memory at 256
rng = random.Random(160)


def call_start(ids):
    return next((i for i, t in enumerate(ids) if t == TC0), None)


with Experiment("EXP-160", "31b-causal-neurons", category="improvement", depends_on=["EXP-147", "EXP-154"],
                motivation="Neurons chosen by correlation with success mark it but are not levers; the author asked for the neurons responsible "
                           "for better performance, which needs an interventional search.",
                hypothesis="Neurons chosen on half the tasks by their attributed effect on 31B's preference for a passing run's edit over its "
                           "own failing edit, when actually scaled, raise that preference in at least 65% of held-out pairs and more than random "
                           "neurons do.",
                config=dict(model=REPO, bench=os.path.relpath(BENCH, ROOT), k=K, alpha=ALPHA, max_call_tokens=MAX_CALL, pairs_per_fail=PAIRS_PER_FAIL)) as ex:
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(BENCH, "rt", encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    pairs = []
    for t, rs in sorted(by.items()):
        good = [r for r in rs if r["solved"] and call_start(r["output_ids"]) is not None]
        bad = [r for r in rs if not r["solved"] and call_start(r["output_ids"]) is not None]
        for f in bad:
            for g in rng.sample(good, min(PAIRS_PER_FAIL, len(good))):
                pairs.append((t, f, g))
    if SMOKE:
        pairs = pairs[:4]
        for _, f, _g in pairs:
            f["prompt_ids"] = f["prompt_ids"][-800:]
    ex.log("pairs", len(pairs), "| tasks", len({p[0] for p in pairs}))
    from transformers import AutoModelForCausalLM
    from sne.chunked_attention import install
    install()
    cache_dir = "/content/hf_" + os.path.basename(REPO.rstrip("/")).lower()
    path = REPO if os.path.isdir(REPO) else (cache_dir if os.path.isdir(cache_dir) else None)
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO, local_dir=cache_dir)
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
    for p_ in model.parameters():
        p_.requires_grad_(False)
    Ls = layers(model)
    nL = len(Ls)
    sizes = [Ls[l].mlp.down_proj.in_features for l in range(nL)]
    st = {"grad": False, "acts": {}, "scale": None}

    def _pre(m, a, l):
        x = a[0]
        if st["scale"] is not None and l in st["scale"]:
            return (x * st["scale"][l],)  # a full-width factor vector: 1 except at the scaled neurons
        if st["grad"]:
            x.retain_grad()
            st["acts"][l] = x
        return None

    hooks = [Ls[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: _pre(m, a, l)) for l in range(nL)]
    emb = model.get_input_embeddings()
    hooks.append(emb.register_forward_hook(lambda m, a, o: o.requires_grad_(True) if st["grad"] else None))

    def mean_logp(prefix, cont, grad):
        """Mean log-probability of cont after prefix. With grad, returns per-layer (activation x gradient) sums over cont."""
        with torch.no_grad():
            out = model(input_ids=torch.tensor([prefix], device=dev), use_cache=True, logits_to_keep=1)
        pkv = out.past_key_values
        first = torch.log_softmax(out.logits[0, -1].float(), -1)[cont[0]]
        st["grad"], st["acts"] = grad, {}
        ctx = torch.enable_grad() if grad else torch.no_grad()
        try:
            with ctx:
                o2 = model(input_ids=torch.tensor([cont], device=dev), past_key_values=pkv, use_cache=True)
                lp = torch.log_softmax(o2.logits[0, :-1].float(), -1)
                tok = torch.tensor(cont[1:], device=dev)
                total = first + lp.gather(1, tok[:, None]).sum()
                m = total / len(cont)
                attr = None
                if grad:
                    m.backward()
                    attr = [(st["acts"][l][0].float() * st["acts"][l].grad[0].float()).sum(0).cpu() for l in range(nL)]
            del o2
        finally:  # an out-of-memory error must not leave gradient capture switched on for the next forward pass
            st["grad"], st["acts"] = False, {}
        del out, pkv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float(m.detach()), attr

    t0 = time.time()
    recs = []
    for n, (t, f, g) in enumerate(pairs):
        cf, cg = call_start(f["output_ids"]), call_start(g["output_ids"])
        prefix = f["prompt_ids"] + f["output_ids"][:cf]
        a = f["output_ids"][cf:cf + MAX_CALL]
        b = g["output_ids"][cg:cg + MAX_CALL]
        try:
            la, aa = mean_logp(prefix, a, True)
            lb, ab = mean_logp(prefix, b, True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            ex.log("pair", n, "skipped: out of memory")
            continue
        recs.append(dict(task=t, prefix=prefix, a=a, b=b, pref=lb - la, attr=torch.cat([ab[l] - aa[l] for l in range(nL)])))
        if n % 5 == 0:
            ex.log(f"pair {n + 1} of {len(pairs)}: preference {lb - la:.3f}, {time.time() - t0:.0f} s so far")
    tasks = sorted({r["task"] for r in recs})
    order = sorted(tasks, key=lambda x: zlib.crc32(x.encode()))
    A, B = order[0::2], order[1::2]
    starts = np.cumsum([0] + sizes)

    def locate(j):
        l = int(np.searchsorted(starts, j, side="right") - 1)
        return l, int(j - starts[l])

    def choose(task_set):
        per = {}
        for r in recs:
            if r["task"] in task_set:
                per.setdefault(r["task"], []).append(r["attr"])
        mean = torch.stack([torch.stack(v).mean(0) for v in per.values()]).mean(0)  # task-balanced
        up = torch.topk(mean, K).indices.tolist()
        down = torch.topk(-mean, K).indices.tolist()
        return up, down, mean

    def scale_map(up, down, alpha):
        sm = {}
        for js, fac in ((up, 1 + alpha), (down, 1 - alpha)):
            for j in js:
                l, i = locate(j)
                sm.setdefault(l, ([], []))
                sm[l][0].append(i)
                sm[l][1].append(fac)
        out = {}
        for l, (ix, fc) in sm.items():
            v = torch.ones(sizes[l])
            for i, f in zip(ix, fc):
                v[i] = f
            out[l] = v.to(device=dev, dtype=torch.bfloat16)
        return out

    def held_out(sel_tasks, test_tasks, alpha):
        up, down, _ = choose(sel_tasks)
        nfeat = int(starts[-1])
        rnd = rng.sample(range(nfeat), 2 * K)
        res = []
        for r in [r for r in recs if r["task"] in test_tasks]:
            row = dict(task=r["task"], base=r["pref"])
            for name, (u, d) in (("chosen", (up, down)), ("random", (rnd[:K], rnd[K:]))):
                st["scale"] = scale_map(u, d, alpha)
                try:
                    la, _ = mean_logp(r["prefix"], r["a"], False)
                    lb, _ = mean_logp(r["prefix"], r["b"], False)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    la = lb = float("nan")
                finally:
                    st["scale"] = None
                row[name] = lb - la
            if all(np.isfinite([row["chosen"], row["random"]])):
                res.append(row)
        return res, up, down

    out = dict(pairs=len(recs), tasks=len(tasks), fold_a=A, fold_b=B, preference_mean=round(float(np.mean([r["pref"] for r in recs])), 4) if recs else None)
    folds = []
    for sel, test in ((A, B), (B, A)):
        if sel and test:
            res, up, down = held_out(sel, test, ALPHA)
            folds += res
    if folds:
        d_ch = np.array([r["chosen"] - r["base"] for r in folds])
        d_rn = np.array([r["random"] - r["base"] for r in folds])
        out["held_out"] = dict(pairs=len(folds), chosen_raises=int((d_ch > 0).sum()), chosen_mean_change=round(float(d_ch.mean()), 4),
                               random_raises=int((d_rn > 0).sum()), random_mean_change=round(float(d_rn.mean()), 4),
                               chosen_minus_random_mean=round(float((d_ch - d_rn).mean()), 4),
                               chosen_beats_random=int((d_ch > d_rn).sum()))
        frac = out["held_out"]["chosen_raises"] / len(folds)
        out["gate_passed"] = bool(frac >= 0.65 and out["held_out"]["chosen_mean_change"] > out["held_out"]["random_mean_change"])
    else:
        out["gate_passed"] = False
    up, down, mean = choose(set(tasks))
    chosen = []
    for j, sgn in [(j, 1) for j in up] + [(j, -1) for j in down]:
        l, i = locate(j)
        chosen.append(dict(layer=l, neuron=i, sign=sgn, attribution=round(float(mean[j]), 6)))
    torch.save(dict(top=chosen, w_down={f"{c['layer']}|{c['neuron']}": Ls[c["layer"]].mlp.down_proj.weight[:, c["neuron"]].float().cpu() for c in chosen},
                    n_inter=max(sizes), layers=nL), os.path.join(OUT, "neurons.pt"))
    out["chosen_layers"] = sorted({c["layer"] for c in chosen})
    out["top5"] = chosen[:5]
    for h in hooks:
        h.remove()
    ex.log("result", {k: v for k, v in out.items() if k not in ("fold_a", "fold_b")})
    ex.save_json("fit.json", out)
    ho = out.get("held_out", {})
    ex.result(headline=f"held out: chosen neurons raise the preference in {ho.get('chosen_raises')} of {ho.get('pairs')} pairs (random {ho.get('random_raises')}); "
                       f"mean change {ho.get('chosen_mean_change')} against {ho.get('random_mean_change')}; gate {'passed' if out['gate_passed'] else 'not passed'}",
              fit=out)

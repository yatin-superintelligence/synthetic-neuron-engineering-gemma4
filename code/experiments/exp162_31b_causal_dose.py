"""EXP-162 (D-224): the causal-neuron search of EXP-160 on all of its pairs, and a dose test. Can a stronger or larger causal edit
close the gap between 31B's failing fix and a passing one, and what does it cost the model's own text?

EXP-160 found 64 neurons that raise 31B's preference for a passing run's edit call over its own failing one on held-out pairs (7 of 7,
random 4 of 7), but only 7 of 38 pairs fit in memory, and the edit closed a quarter of the gap (0.123 of 0.456 nats per token); in the
agent it solved 9 of 15 (EXP-161a, unedited 9 to 13).

Changes from EXP-160: the prompt is prefilled in chunks of 1,024 tokens through the cache (and attention queries in chunks of 512), with
expandable CUDA segments, so long prompts fit; one prefill per configuration serves both continuations (the cache is copied). Pairs
that still run out of memory are retried with the prompt cut to its first 1,024 and last 7,168 tokens (flagged).
Stage 1: attribution on every pair (as EXP-160). Stage 2, held out by task (two folds, EXP-160's split rule): for neurons chosen on the
other fold, the preference and the damage under each configuration:
  chosen K=32 and K=128 per side, amplified side x2 and x4 (the damped side zeroed); random neurons at K=32 x2 and K=128 x4.
Damage: the drop in mean log-probability of the last 128 tokens of the failing run's own thought (text it wrote), nats per token.
Flip: the edited preference is above 0 (the passing call is more likely than the failing one).
Outputs: experiments/EXP-162_*/results.json, artifacts_local/EXP-162/{dose.json, neurons.pt}. Runs on a Colab G4 (bf16 31B)."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import copy  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE  # noqa: E402
from sne.model import layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("SCAN_MODEL") or (os.path.join(ROOT, "models", "gemma-4-E2B-it-qat-unquantized") if SMOKE else "google/gemma-4-31B-it-qat-q4_0-unquantized")
BENCH = os.environ.get("SCAN_BENCH") or os.path.join(ROOT, "artifacts_local", "EXP-147", "success_bench.jsonl.gz")
EXP_ID = os.environ.get("EXP162_ID", "EXP-162")  # EXP-162b: every prompt cut up front (D-225)
CUT_ALL = os.environ.get("EXP162_CUT_ALL") == "1"
OUT = os.path.join(ROOT, "scratch", "preflight", EXP_ID) if SMOKE else os.path.join(ROOT, "artifacts_local", EXP_ID)
TC0 = 48
MAX_CALL, PAIRS_PER_FAIL, W = (40, 1, 16) if SMOKE else (64, 1, 128)
PREFILL, HEAD, TAIL = (256, 128, 512) if SMOKE else (1024, 1024, int(os.environ.get("EXP162_TAIL", 7168)))
CONFIGS = [("chosen", 32, 1.0), ("chosen", 32, 3.0), ("chosen", 128, 1.0), ("chosen", 128, 3.0), ("random", 32, 1.0), ("random", 128, 3.0)]
if SMOKE:
    CONFIGS = [("chosen", 4, 1.0), ("random", 4, 1.0)]
STAGE2_LIMIT_S = 35 * 60
rng = random.Random(160)  # EXP-160's seed: the same pairing, so its 7 pairs are among these


def call_start(ids):
    return next((i for i, t in enumerate(ids) if t == TC0), None)


with Experiment(EXP_ID, "31b-causal-dose" + ("-all-cut" if CUT_ALL else ""), category="improvement", depends_on=["EXP-160", "EXP-161a"],
                motivation="EXP-160's causal neurons moved 31B a quarter of the way to the passing fix on 7 pairs and did not raise tasks "
                           "solved; the open questions are whether the effect holds on all pairs and whether a stronger edit closes the gap.",
                hypothesis="On all pairs, the K=32 x2 edit chosen on the other fold raises the held-out preference in at least 65% of pairs "
                           "and more than random neurons. Prediction for the dose: no configuration with damage of 0.10 nats per token or "
                           "less flips the preference in a quarter more of the held-out pairs than are already flipped unedited.",
                config=dict(model=REPO, bench=os.path.relpath(BENCH, ROOT), max_call_tokens=MAX_CALL, pairs_per_fail=PAIRS_PER_FAIL,
                            own_window=W, prefill_chunk=PREFILL, truncate_to=[HEAD, TAIL], configs=CONFIGS)) as ex:
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
            f["prompt_ids"] = f["prompt_ids"][-700:]
    ex.log("pairs", len(pairs), "| tasks", len({p[0] for p in pairs}))
    from transformers import AutoModelForCausalLM
    from sne.chunked_attention import install
    install(chunk=512, min_len=512)
    cache_dir = "/content/hf_" + os.path.basename(REPO.rstrip("/")).lower()
    path = REPO if os.path.isdir(REPO) else (cache_dir if os.path.isdir(cache_dir) else None)
    if path is None:
        from huggingface_hub import snapshot_download
        t_dl = time.time()
        path = snapshot_download(REPO, local_dir=cache_dir)
        ex.log(f"downloaded in {time.time() - t_dl:.0f} s")
    t_ld = time.time()
    if torch.cuda.is_available():
        dev = "cuda"
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True).eval()
        for nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
            if hasattr(model.model, nm) and getattr(model.model, nm) is not None:
                setattr(model.model, nm, None)
        torch.cuda.empty_cache()
    else:
        from sne.model import device as _device, load as _load
        model, _ = _load(path)
        dev = _device()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    Ls = layers(model)
    nL = len(Ls)
    sizes = [Ls[l].mlp.down_proj.in_features for l in range(nL)]
    ex.log(f"model loaded in {time.time() - t_ld:.0f} s | layers {nL} | neurons {sum(sizes)}"
           + (f" | GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB" if torch.cuda.is_available() else ""))
    st = {"grad": False, "acts": {}, "scale": None}

    def _pre(m, a, l):
        x = a[0]
        if st["scale"] is not None and l in st["scale"]:
            return (x * st["scale"][l],)
        if st["grad"]:
            x.retain_grad()
            st["acts"][l] = x
        return None

    hooks = [Ls[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: _pre(m, a, l)) for l in range(nL)]
    emb = model.get_input_embeddings()
    hooks.append(emb.register_forward_hook(lambda m, a, o: o.requires_grad_(True) if st["grad"] else None))

    def prefill(prefix):
        """Chunked prefill. Returns the cache, the log-distribution for the first continuation token, and the mean log-probability
        of the last W prefix tokens (the failing run's own text)."""
        n = len(prefix)
        bounds = list(range(0, n, PREFILL)) + [n]
        if len(bounds) > 2 and n - bounds[-2] < W + 1:
            bounds.pop(-2)
        pkv, out = None, None
        with torch.no_grad():
            for s, e in zip(bounds[:-1], bounds[1:]):
                keep = W + 1 if e == n else 1
                out = model(input_ids=torch.tensor([prefix[s:e]], device=dev), past_key_values=pkv, use_cache=True, logits_to_keep=keep)
                pkv = out.past_key_values
            lg = out.logits[0].float()
            first = torch.log_softmax(lg[-1], -1)
            lp = torch.log_softmax(lg[:-1], -1)
            own = float(lp.gather(1, torch.tensor(prefix[n - W:], device=dev)[:, None]).mean())
        del out, lg, lp
        return pkv, first, own

    def cont_logp(pkv, first, cont, grad):
        st["grad"], st["acts"] = grad, {}
        try:
            with (torch.enable_grad() if grad else torch.no_grad()):
                o2 = model(input_ids=torch.tensor([cont], device=dev), past_key_values=pkv, use_cache=True)
                lp = torch.log_softmax(o2.logits[0, :-1].float(), -1)
                m = (first[cont[0]] + lp.gather(1, torch.tensor(cont[1:], device=dev)[:, None]).sum()) / len(cont)
                attr = None
                if grad:
                    m.backward()
                    attr = [(st["acts"][l][0].float() * st["acts"][l].grad[0].float()).sum(0).cpu() for l in range(nL)]
                del o2, lp
        finally:
            st["grad"], st["acts"] = False, {}
        return float(m.detach()), attr

    def evaluate(prefix, a, b, grad):
        """Preference (mean log p of b minus a), own-text log p, and (with grad) the attribution b minus a."""
        pkv, first, own = prefill(prefix)
        pkv_b = copy.deepcopy(pkv)
        la, aa = cont_logp(pkv, first, a, grad)
        del pkv
        lb, ab = cont_logp(pkv_b, first, b, grad)
        del pkv_b, first
        attr = torch.cat([ab[l] - aa[l] for l in range(nL)]) if grad else None
        return lb - la, own, attr

    def run_pair(prefix, a, b, grad):
        try:
            return evaluate(prefix, a, b, grad), False
        except torch.OutOfMemoryError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(prefix) <= HEAD + TAIL:
            return None, False
        try:
            return evaluate(prefix[:HEAD] + prefix[-TAIL:], a, b, grad), True
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, True

    t0 = time.time()
    recs, skipped = [], 0
    order_by_len = sorted(range(len(pairs)), key=lambda n: len(pairs[n][1]["prompt_ids"]))
    for k, n in enumerate(order_by_len):
        t, f, g = pairs[n]
        cf, cg = call_start(f["output_ids"]), call_start(g["output_ids"])
        prefix = f["prompt_ids"] + f["output_ids"][:cf]
        a, b = f["output_ids"][cf:cf + MAX_CALL], g["output_ids"][cg:cg + MAX_CALL]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if CUT_ALL and len(prefix) > HEAD + TAIL:  # memory grows about 4.5 MB per prompt token on the G4 (EXP-162): cut before trying
            try:
                res, cut = evaluate(prefix[:HEAD] + prefix[-TAIL:], a, b, True), True
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                res, cut = None, True
        else:
            res, cut = run_pair(prefix, a, b, True)
        if res is None:
            skipped += 1
            ex.log(f"pair {n} ({len(prefix)} tokens) skipped: out of memory")
            continue
        pref, own, attr = res
        used = prefix[:HEAD] + prefix[-TAIL:] if cut else prefix
        recs.append(dict(task=t, n=n, prefix=used, a=a, b=b, pref=pref, own=own, attr=attr, truncated=cut, tokens=len(prefix)))
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        ex.log(f"pair {k + 1} of {len(pairs)}: {len(prefix)} tokens{' (cut)' if cut else ''}, preference {pref:.3f}, peak {peak:.1f} GB, "
               f"{time.time() - t0:.0f} s so far")
    tasks = sorted({r["task"] for r in recs})
    order = sorted(tasks, key=lambda x: zlib.crc32(x.encode()))
    A, B = order[0::2], order[1::2]
    starts = np.cumsum([0] + sizes)
    nfeat = int(starts[-1])
    ex.log(f"stage 1 done: {len(recs)} pairs on {len(tasks)} tasks ({sum(r['truncated'] for r in recs)} cut, {skipped} skipped); "
           f"mean preference {np.mean([r['pref'] for r in recs]):.3f}")

    def locate(j):
        l = int(np.searchsorted(starts, j, side="right") - 1)
        return l, int(j - starts[l])

    def choose(task_set, k):
        per = {}
        for r in recs:
            if r["task"] in task_set:
                per.setdefault(r["task"], []).append(r["attr"])
        mean = torch.stack([torch.stack(v).mean(0) for v in per.values()]).mean(0)
        return torch.topk(mean, k).indices.tolist(), torch.topk(-mean, k).indices.tolist(), mean

    def scale_map(up, down, alpha):
        out = {}
        for js, fac in ((up, 1 + alpha), (down, 0.0)):
            for j in js:
                l, i = locate(j)
                if l not in out:
                    out[l] = torch.ones(sizes[l])
                out[l][i] = fac
        return {l: v.to(device=dev, dtype=torch.bfloat16) for l, v in out.items()}

    rnd_pool = rng.sample(range(nfeat), 2 * max(k for _, k, _ in CONFIGS))
    rows_out = []
    t1 = time.time()
    for sel, test in ((A, B), (B, A)):
        if not sel or not test:
            continue
        chosen = {k: choose(sel, k)[:2] for k in sorted({k for _, k, _ in CONFIGS})}
        for r in [r for r in recs if r["task"] in test]:
            if time.time() - t1 > STAGE2_LIMIT_S:
                break
            row = dict(task=r["task"], n=r["n"], tokens=r["tokens"], truncated=r["truncated"], base=r["pref"], own=r["own"])
            for kind, k, alpha in CONFIGS:
                up, down = chosen[k] if kind == "chosen" else (rnd_pool[:k], rnd_pool[k:2 * k])
                st["scale"] = scale_map(up, down, alpha)
                try:
                    pref, own, _ = evaluate(r["prefix"], r["a"], r["b"], False)
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pref = own = float("nan")
                finally:
                    st["scale"] = None
                row[f"{kind}_{k}_x{int(1 + alpha)}"] = dict(pref=pref, damage=r["own"] - own)
            rows_out.append(row)
            ex.log(f"held-out pair {len(rows_out)}: base {row['base']:.3f} | "
                   + " ".join(f"{c}: {v['pref']:.3f}/{v['damage']:.3f}" for c, v in row.items() if isinstance(v, dict))
                   + f" | {time.time() - t1:.0f} s")
    summary = {}
    for kind, k, alpha in CONFIGS:
        c = f"{kind}_{k}_x{int(1 + alpha)}"
        ok = [r for r in rows_out if np.isfinite(r[c]["pref"])]
        if not ok:
            continue
        d = np.array([r[c]["pref"] - r["base"] for r in ok])
        summary[c] = dict(pairs=len(ok), raises=int((d > 0).sum()), mean_change=round(float(d.mean()), 4),
                          flips=int(sum(r[c]["pref"] > 0 for r in ok)), base_flips=int(sum(r["base"] > 0 for r in ok)),
                          mean_damage=round(float(np.mean([r[c]["damage"] for r in ok])), 4),
                          mean_gap=round(float(np.mean([r["base"] for r in ok])), 4))
    out = dict(pairs=len(recs), tasks=len(tasks), truncated=sum(r["truncated"] for r in recs), skipped=skipped,
               preference_mean=round(float(np.mean([r["pref"] for r in recs])), 4) if recs else None, held_out_pairs=len(rows_out),
               summary=summary)
    rep = summary.get("chosen_32_x2"), summary.get("random_32_x2")
    if rep[0] and rep[1]:
        out["replication_passed"] = bool(rep[0]["raises"] >= 0.65 * rep[0]["pairs"] and rep[0]["mean_change"] > rep[1]["mean_change"])
    leads = [c for c, v in summary.items() if c.startswith("chosen") and v["mean_damage"] <= 0.10 and (v["flips"] - v["base_flips"]) >= 0.25 * v["pairs"]]
    out["dose_leads"] = leads
    best = max((c for c in summary if c.startswith("chosen")), key=lambda c: summary[c]["mean_change"], default=None)
    if best:
        _, k, x = best.split("_")
        up, down, mean = choose(set(tasks), int(k))
        top = []
        for j, sgn in [(j, 1) for j in up] + [(j, -1) for j in down]:
            l, i = locate(j)
            top.append(dict(layer=l, neuron=i, sign=sgn, attribution=round(float(mean[j]), 6)))
        torch.save(dict(top=top, factor_up=float(x[1:]), config=best, n_inter=max(sizes), layers=nL,
                        w_down={f"{c['layer']}|{c['neuron']}": Ls[c["layer"]].mlp.down_proj.weight[:, c["neuron"]].float().cpu() for c in top}),
                   os.path.join(OUT, "neurons.pt"))
        out["saved_config"] = best
    for h in hooks:
        h.remove()
    json.dump(dict(out, rows=rows_out), open(os.path.join(OUT, "dose.json"), "w"), indent=1)
    ex.log("result", out)
    s32, r32 = summary.get("chosen_32_x2", {}), summary.get("random_32_x2", {})
    ex.result(headline=f"{len(recs)} pairs on {len(tasks)} tasks; held out, chosen K=32 x2 raises the preference in {s32.get('raises')} of "
                       f"{s32.get('pairs')} (random {r32.get('raises')}); dose leads: {leads or 'none'}", fit=out)

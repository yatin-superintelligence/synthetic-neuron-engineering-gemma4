"""EXP-095: copy-head gain for E2B's edit_file calls, training-free.

Finding behind it (2026-11-08, five unedited official runs, 584 edit_file calls): the largest group of failed edits (174 of 364)
has an old_string line that is not in the file the model read; many are older versions of the code recalled from memory (for
example an earlier FastAPI signature). Writing old_string is a copy from the context. E2B's sliding layers see only the last 512
tokens, so a copy from a file read earlier runs through its 7 full-attention layers and through whichever heads attend to the
token being copied.

1. Copy scores (bf16, teacher-forced on EXP-092's fit samples whose old_string applies). For each old_string token, its source is
   the prompt position with the longest unique backward match (at least 2 tokens). A head's copy score is its mean attention from
   the position that predicts the token to the source token, over all such tokens (all 35 layers x 8 heads).
2. Edit: head h of layer l gets gain beta on its output's main subspace. c_h = W_o[:, cols_h] z_h is the head's write. U_r holds the
   top r = 64 principal directions of c_h on the copy rows. o_proj += (beta - 1) U_r U_r^T W_o[:, cols_h] E_h, an exact LoRA of rank
   64 per head (at most 2 heads per layer, so at most rank 128). The share of c_h's energy in U_r is reported.

Pre-registered (D-070, before any result):
  Heads: the k with the highest copy scores (at most 2 per layer). Arms: k in {4, 8} x beta in {2, 3}.
  Selection on EXP-092's fit split (257 states): the most applying answers (ties: fewer heads, smaller beta).
  Test on EXP-092's dev and test splits together (257 states): unedited, the selected edit, and a control edit (the same number of
    heads, drawn at random among heads with a copy score below the median, seed 95; same beta and r).
  Retention: MBPP+ all 378 problems (cap 8,192 tokens), unedited, selected and control.
  Gate for an official agent run (seed 1): test applies above the unedited model (exact McNemar two-sided p < 0.05) and above the
    control; correct answers not fewer; MBPP+ not lower with McNemar p < 0.05.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import ROOT, SMOKE, bench  # noqa: E402
from sne.registry import Experiment  # noqa: E402

HERE = os.path.dirname(__file__)
_src = open(os.path.join(HERE, "exp092_patch_neurons.py"), encoding="utf-8").read()
P = {"__file__": os.path.join(HERE, "exp092_patch_neurons.py"), "__name__": "exp092"}
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)
_src94 = open(os.path.join(HERE, "exp094_bos_value_edit.py"), encoding="utf-8").read()
Q = {"__file__": os.path.join(HERE, "exp094_bos_value_edit.py"), "__name__": "exp094"}
exec(compile(_src94.split('with Experiment("EXP-094"')[0], "exp094", "exec"), Q)  # Suite
B92 = os.path.join(ROOT, "artifacts_local", "EXP-092")
OUT = os.path.join(ROOT, "artifacts_local", "EXP-095")
R = 64
ARMS = [(4, 2.0), (4, 3.0), (8, 2.0), (8, 3.0)]
STR = __import__("re").compile(r'old_string:<\|"\|>(.*?)<\|"\|>', __import__("re").S)


def copy_rows(tok, p_ids, text):
    """(query position, source position) for old_string tokens with a unique backward match of at least 2 tokens in the prompt."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    a_ids, offs = enc["input_ids"], enc["offset_mapping"]
    m = STR.search(text, max(0, text.find("<|tool_call>")))
    if not m:
        return a_ids, []
    s0, s1 = m.span(1)
    full = p_ids + a_ids
    n = len(p_ids)
    where = {}
    for i, t in enumerate(p_ids):
        where.setdefault(t, []).append(i)
    rows = []
    for j, (a, b) in enumerate(offs):
        if not (s0 <= a and b <= s1 and b > a):
            continue
        t = n + j
        best, blen, tie = None, 0, False
        for s in where.get(full[t], []):
            k = 0
            while k < 12 and s - 1 - k >= 0 and full[s - 1 - k] == full[t - 1 - k]:
                k += 1
            if k > blen:
                best, blen, tie = s, k, False
            elif k == blen:
                tie = True
        if best is not None and blen >= 2 and not tie:
            rows.append((t - 1, best))
    return a_ids, rows


def scores(tok, samples, ex):
    from huggingface_hub import snapshot_download
    import transformers.models.gemma4.modeling_gemma4 as mg
    from sne.model import layers, load
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False, attn_implementation="sdpa")
    Ls = layers(model)
    nL, H = len(Ls), Ls[0].self_attn.config.num_attention_heads
    CAP = dict(rows=None, src=None)
    mass = torch.zeros(nL, H, dtype=torch.float64)
    hits = torch.zeros(nL, H, dtype=torch.float64)
    cnt = [0]
    orig = mg.ALL_ATTENTION_FUNCTIONS["sdpa"]

    def wrapped(module, query, key, value, attention_mask, **kw):
        out = orig(module, query, key, value, attention_mask, **kw)
        if CAP["rows"] is not None:
            rows, src = CAP["rows"], CAP["src"]
            q = query[0][:, rows].float()  # H x n x D
            k = key[0][0].float() if key.shape[1] == 1 else None
            if k is None:
                raise RuntimeError("expected one key/value head")
            lg = (q @ k.T) * (kw.get("scaling") or 1.0)  # H x n x S
            S = k.shape[0]
            j = torch.arange(S, device=lg.device)[None, :]
            ok = j <= rows[:, None]
            w = kw.get("sliding_window")
            if w:
                ok = ok & (rows[:, None] - j < w)
            lg = lg.masked_fill(~ok[None], float("-inf"))
            p = torch.softmax(lg, -1)
            ar = torch.arange(len(rows), device=lg.device)
            L = module.layer_idx
            mass[L] += p[:, ar, src].sum(-1).double().cpu()
            hits[L] += (p.argmax(-1) == src[None]).sum(-1).double().cpu()
        return out
    mg.ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
    Z = {L: [] for L in range(nL)}
    hk = [Ls[L].self_attn.o_proj.register_forward_pre_hook(lambda m, a, L=L: Z[L].append(a[0][0][CAP["rows"]].half().cpu()) if CAP["rows"] is not None else None)
          for L in range(nL)]
    used = 0
    with torch.no_grad():
        for k, (s, a) in enumerate(samples):
            prompt = P["G"]["render"](tok, s["prefix"], True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids, rows = copy_rows(tok, p_ids, a["text"])
            if len(rows) < 3:
                continue
            CAP["rows"] = torch.tensor([r for r, _ in rows], device="cuda")
            CAP["src"] = torch.tensor([s_ for _, s_ in rows], device="cuda")
            model(input_ids=torch.tensor([p_ids + a_ids], device="cuda"), use_cache=False, logits_to_keep=1)
            cnt[0] += len(rows)
            used += 1
            CAP["rows"] = None
            if k % 20 == 0:
                print(f"[copy] sample {k}/{len(samples)} rows so far {cnt[0]}", flush=True)
    for h in hk:
        h.remove()
    mg.ALL_ATTENTION_FUNCTIONS["sdpa"] = orig
    sc = (mass / max(1, cnt[0])).float()
    hr = (hits / max(1, cnt[0])).float()
    W = {L: Ls[L].self_attn.o_proj.weight.float().cpu() for L in range(nL)}
    hd = {L: Ls[L].self_attn.head_dim for L in range(nL)}
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    rep = dict(samples_used=used, copy_rows=cnt[0], top=[], layer_types={L: ("full" if hd[L] == 512 else "sliding") for L in range(nL)})
    order = sorted(((float(sc[L, h]), L, h) for L in range(nL) for h in range(H)), reverse=True)
    rep["top"] = [dict(layer=L, head=h, score=round(v, 4), argmax_hit=round(float(hr[L, h]), 4)) for v, L, h in order[:16]]
    rep["median_score"] = round(float(sc.median()), 5)
    ex.log("copy heads, top 16:", json.dumps(rep["top"]))
    Zc = {L: torch.cat(Z[L]).float() if Z[L] else None for L in range(nL)}
    return sc, order, Zc, W, hd, rep


def head_lora(Zc, W, hd, heads, beta, path):
    """o_proj += (beta - 1) U_r U_r^T W_o[:, cols_h] E_h for each (layer, head); returns the path and the energy shares."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    by = {}
    for L, h in heads:
        by.setdefault(L, []).append(h)
    sd, share, maxr = {}, {}, 0
    for L, hs in sorted(by.items()):
        d = hd[L]
        A_rows, B_cols = [], []
        for h in hs:
            cols = slice(h * d, (h + 1) * d)
            Wh = W[L][:, cols]  # hidden x d
            Ztr = Zc[L][:, cols]
            n = len(Ztr)
            cut = max(1, int(n * 0.8))
            C = Ztr[:cut] @ Wh.T  # rows x hidden
            ev, U = torch.linalg.eigh(C.T @ C)
            Ur = U[:, -R:]  # hidden x R
            Ch = Ztr[cut:] @ Wh.T
            share[f"L{L}h{h}"] = round(float((Ch @ Ur).pow(2).sum() / Ch.pow(2).sum().clamp_min(1e-12)), 4) if len(Ch) else None
            A = torch.zeros(R, W[L].shape[1])
            A[:, cols] = Ur.T @ Wh
            A_rows.append(A)
            B_cols.append((beta - 1.0) * Ur)
        sd[L] = (torch.cat(A_rows), torch.cat(B_cols, 1))
        maxr = max(maxr, sd[L][0].shape[0])
    out = {}
    for L, (A, B) in sd.items():  # every module zero-padded to one rank, so alpha / r = 1 holds for all of them in vLLM and PEFT
        pad = maxr - A.shape[0]
        A = torch.cat([A, torch.zeros(pad, A.shape[1])]) if pad else A
        B = torch.cat([B, torch.zeros(B.shape[0], pad)], 1) if pad else B
        base = f"base_model.model.model.language_model.layers.{L}.self_attn.o_proj"
        out[f"{base}.lora_A.weight"] = A.contiguous()
        out[f"{base}.lora_B.weight"] = B.contiguous()
    save_file(out, os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=maxr, lora_alpha=maxr,
                   target_modules=["o_proj"], layers_to_transform=sorted(by), lora_dropout=0.0, bias="none", fan_in_fan_out=False,
                   init_lora_weights=True, inference_mode=True, use_rslora=False),
              open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path, share, maxr


def pick(order, k, exclude=()):
    out, per = [], {}
    for _, L, h in order:
        if (L, h) in exclude or per.get(L, 0) >= 2:
            continue
        out.append((L, h))
        per[L] = per.get(L, 0) + 1
        if len(out) == k:
            break
    return out


class RankServer:
    """EXP-092's server with a LoRA rank limit large enough for the head edits."""

    def __init__(self, loras, log, rank):
        import subprocess
        import time
        import urllib.request
        srv = P["Server"].__new__(P["Server"])
        md = P["official_model"]()
        r = next(x for x in (8, 16, 32, 64, 128, 256) if x >= rank)
        cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--host", "127.0.0.1",
               "--port", str(P["PORT"]), "--max-model-len", "32768", "--gpu-memory-utilization", "0.80", "--enforce-eager", "--max-num-seqs", "128",
               "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0}), "--enable-lora", "--max-lora-rank", str(r), "--max-loras", str(len(loras)),
               "--max-cpu-loras", str(len(loras)), "--lora-modules", *[f"{n}={p}" for n, p in loras.items()]]
        srv.p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 1200:
            if srv.p.poll() is not None:
                raise RuntimeError("vLLM exited: " + open(log, errors="replace").read()[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{P['PORT']}/health", timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(5)
        else:
            raise RuntimeError("vLLM not ready in 20 min")
        self.srv = srv

    def __getattr__(self, name):
        return getattr(self.srv, name)


with Experiment("EXP-095", "copy-head-gain", category="improvement", depends_on=["EXP-092"],
                motivation="The largest group of E2B's failed edits writes old_string lines that are not in the file it read (174 of 364 in five unedited "
                           "official runs). A gain on the heads that copy old_string from the context targets that step without training.",
                hypothesis="Copy-head gain raises the share of applying bench answers on held-out states against the unedited model (exact McNemar p < 0.05) "
                           "and against a control gain on low-scoring heads, with no fewer correct answers and no significant MBPP+ loss.",
                config=dict(rank_per_head=R, arms=ARMS, selection="fit split, most applying", test="dev + test splits (257)",
                            retention="MBPP+ 378, cap 8,192", control="same count of random heads below the median copy score, seed 95")) as ex:
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:24]
    states, _ = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    rng = random.Random(92)  # EXP-092's split
    ids = sorted({s["iid"] for s in states})
    rng.shuffle(ids)
    n = len(ids)
    split = {i: ("fit" if k < n // 2 else "dev" if k < 3 * n // 4 else "test") for k, i in enumerate(ids)}
    for s in states:
        s["split"] = split[s["iid"]]

    def vprompt(prefix):
        p = P["G"]["render"](tok, prefix, True)
        return p[len("<bos>"):] if p.startswith("<bos>") else p
    prompts = {s["iid"]: vprompt(s["prefix"]) for s in states}
    base = json.load(open(os.path.join(B92, "base_greedy.json"), encoding="utf-8"))
    saved = json.load(open(os.path.join(B92, "fit_samples.json"), encoding="utf-8"))
    by_id = {s["iid"]: s for s in states}
    samples = [(by_id[a["iid"]], a) for a in saved if a["iid"] in by_id and by_id[a["iid"]]["split"] == "fit" and a["applies"]]
    random.Random(95).shuffle(samples)
    samples = samples[: (12 if SMOKE else 140)]
    fit = [s for s in states if s["split"] == "fit"]
    held = [s for s in states if s["split"] in ("dev", "test")]
    sc, order, Zc, W, hd, rep = scores(tok, samples, ex)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    loras, shares, maxr = {}, {}, 0
    for k, beta in ARMS:
        nm = f"h{k}_b{int(beta)}"
        heads = pick(order, k)
        loras[nm], shares[nm], r = head_lora(Zc, W, hd, heads, beta, os.path.join(OUT, nm))
        maxr = max(maxr, r)
        rep[nm] = dict(heads=heads, energy_share=shares[nm])
    ex.save_json("heads.json", rep)

    def run(srv, name, ss):
        outs = srv.answers(name, [prompts[s["iid"]] for s in ss])
        return [P["score"](o[0], s) for s, o in zip(ss, outs)]

    def all_arms(srv, names, ss):  # arms in parallel, so vLLM's 128 slots stay full through each arm's slow tail
        from concurrent.futures import ThreadPoolExecutor as TP
        with TP(len(names)) as tp:
            return dict(zip(names, tp.map(lambda k: run(srv, k, ss), names)))
    srv = RankServer(loras, os.path.join(OUT, "vllm_scan.log"), maxr)
    try:
        scan = all_arms(srv, list(loras) + ["base"], fit)  # D-071: unedited re-run in the same server
    finally:
        srv.close()
    counts = {k: dict(applies=sum(r["applies"] for r in v), correct=sum(r["correct"] for r in v)) for k, v in scan.items()}
    counts["base_stored"] = dict(applies=sum(base[s["iid"]]["applies"] for s in fit), correct=sum(base[s["iid"]]["correct"] for s in fit))
    ex.log("fit-split scan (of", len(fit), "):", counts)
    best = max(list(loras), key=lambda k: (counts[k]["applies"], -int(k.split("_b")[0][1:]), -int(k.split("_b")[1])))
    kb, bb = int(best.split("_b")[0][1:]), float(best.split("_b")[1])
    low = [(L, h) for v, L, h in order if v < float(sc.median())]
    random.Random(95).shuffle(low)
    ctrl_heads = pick([(0.0, L, h) for L, h in low], kb)
    ctrl, cshare, cr = head_lora(Zc, W, hd, ctrl_heads, bb, os.path.join(OUT, "control"))
    rep["control"] = dict(heads=ctrl_heads, energy_share=cshare)
    ex.save_json("heads.json", rep)
    srv = RankServer({"selected": loras[best], "control": ctrl}, os.path.join(OUT, "vllm_test.log"), max(maxr, cr))
    try:
        res = all_arms(srv, ["base", "selected", "control"], held)
        mb = bench.mbppplus("dev") + bench.mbppplus("test")
        if SMOKE:
            mb = mb[:6]
        from sne.evaluate import run_suite
        mbpp = {}
        for name in ("base", "selected", "control"):
            summ, rws = run_suite(Q["Suite"](srv, name), tok, mb, max_tokens=8192)
            mbpp[name] = {r["task_id"]: r["passed"] for r in rws}
            ex.log(f"[copy] MBPP+ {name}: {summ['passed']}/{summ['n']}")
    finally:
        srv.close()
    ba = [r["applies"] for r in res["base"]]  # D-071: the unedited arm of the same server run
    bc = [r["correct"] for r in res["base"]]
    sa, scc = [r["applies"] for r in res["selected"]], [r["correct"] for r in res["selected"]]
    ca, cc = [r["applies"] for r in res["control"]], [r["correct"] for r in res["control"]]
    x, y, p = P["mcnemar"](sa, ba)
    tid = [t["task_id"] for t in mb]
    mx, my, mp = P["mcnemar"]([mbpp["selected"][i] for i in tid], [mbpp["base"][i] for i in tid])
    gate = p < 0.05 and x > y and sum(sa) > sum(ca) and sum(scc) >= sum(bc) and not (my > mx and mp < 0.05)
    ex.result(heads=rep, fit_scan=counts, selected=best,
              test=dict(n=len(held), applies=dict(base=sum(ba), selected=sum(sa), control=sum(ca), selected_only=x, base_only=y, p=round(p, 4)),
                        correct=dict(base=sum(bc), selected=sum(scc), control=sum(cc))),
              mbpp=dict(n=len(tid), base=sum(mbpp["base"].values()), selected=sum(mbpp["selected"].values()), control=sum(mbpp["control"].values()),
                        selected_only=mx, base_only=my, p=round(mp, 4)),
              gate_passed=gate,
              headline=f"copy-head gain {best}: test applies unedited {sum(ba)}/{len(held)}, selected {sum(sa)} (control {sum(ca)}), McNemar p = {round(p, 4)}; "
                       f"correct {sum(bc)} -> {sum(scc)}; MBPP+ {sum(mbpp['base'].values())} -> {sum(mbpp['selected'].values())} of {len(tid)}; "
                       f"gate {'passed' if gate else 'not passed'}")

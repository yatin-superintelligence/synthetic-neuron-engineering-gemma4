"""EXP-098: file-over-thought attention steering for E2B's edit_file calls, training-free.

Finding behind it (2026-11-09, EXP-092's bench answers): when E2B's edit fails because an old_string line is not in the file, that
line appears verbatim in E2B's own thought, earlier in the same answer, in 31 of 66 greedy failures (47%) and 129 of 183 sampled
failures (70%). E2B paraphrases the code while thinking, then copies old_string from its nearby thought instead of from the file it
read. Its copy heads sit in the full-attention layers (EXP-095: 24, 29 and 34 hold 13 of the 16 top heads), whose keys and values
come from layer 14. In those layers proportional RoPE rotates only 25% of each head's 512 dimensions; the other 384 carry no position,
so a query component there shifts attention by content, at any distance.

The edit, at each chosen head h of layer L in {24, 29, 34}: d is the unit vector of mean(key at file tokens) - mean(key at thought
tokens) in the position-free dimensions (layer 14's normed, rotated keys, teacher-forced on fit-split answers). The query gains c d
after its norm (q_proj adds c rms_q (d / gamma_q) before q_norm), which raises every file token's logit against every thought token's
by c |k_file - k_thought| on average. It acts only where E2B writes a tool-call string: q_proj += B v^T, with v a ridge read-out on
q_proj's input (about 1 at tool-call string positions, about 0 on thought and prompt) and B the per-head shifts: an exact rank-1
q_proj LoRA per layer.

Pre-registered (D-078, before any result):
  Heads (EXP-095's copy-score order in layers 24, 29, 34): top 4, or top 8. Shift: the file-versus-thought logit gap raised by 1.5 or
    4. Four edits; the measured change in the heads' attention share on the file (against the thought) at copy rows is reported.
  Selection on EXP-092's fit split (257 states) by applying answers, with the unedited arm in the same server (D-071).
  Test on dev and test (257): unedited, selected, and a control (a random unit direction in the same position-free dimensions,
    orthogonal to d, same heads and size), one server. Also counted: failing answers whose not-in-file lines all appear in the
    thought (self-copies).
  Retention: MBPP+ all 378 problems (cap 8,192 tokens), unedited, selected and control.
  Gate for an official agent run (seed 1): test applies above the unedited arm (exact McNemar two-sided p < 0.05) and above the
    control; correct answers not fewer; MBPP+ not lower with McNemar p < 0.05.
"""
import json
import os
import random
import re
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
OUT = os.path.join(ROOT, "artifacts_local", "EXP-098")
COPY_HEADS = [(24, 1), (29, 4), (29, 5), (29, 1), (29, 0), (24, 3), (34, 0), (24, 0), (29, 3), (34, 5), (34, 7), (29, 7), (34, 6)]  # EXP-095 log
HEADSETS = {"top4": COPY_HEADS[:4], "top8": COPY_HEADS[:8]}
GAPS = [1.5, 4.0]
LAYERS = sorted({L for L, _ in COPY_HEADS[:8]})
STR = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.S)
OLD = re.compile(r'old_string:<\|"\|>(.*?)<\|"\|>', re.S)


def nope_mask(D):
    m = torch.ones(D)
    r = D // 8  # 25% of the dimensions rotate: 64 angle pairs of 256, laid out as [0, 64) and [256, 320) by rotate_half
    m[:r] = 0
    m[D // 2:D // 2 + r] = 0
    return m


def regions(tok, prompt, p_ids, text, excerpt):
    """Token positions of the file content (last read_file), of the thought, of tool-call string writing, and old_string copy rows."""
    n = len(p_ids)
    pe = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    body = json.dumps(excerpt)[1:-1]
    a, ln = prompt.rfind(body[:300]), len(body)
    if a < 0:  # the template may show the tool output without JSON escaping
        a, ln = prompt.rfind(excerpt[:300]), len(excerpt)
    file_pos = []
    if a >= 0:
        b = a + ln
        file_pos = [i for i, (s, e) in enumerate(pe) if s >= a and e <= b and e > s]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    tc = text.find("<|tool_call>")
    if tc < 0:
        return ids, file_pos, [], [], []
    spans = [m.span(1) for m in STR.finditer(text, tc)]
    mo = OLD.search(text, tc)
    o0, o1 = mo.span(1) if mo else (-1, -1)
    thought, inside, rows = [], [], []
    for j, (s, e) in enumerate(offs):
        if e <= tc:
            thought.append(n + j)
        elif any(x <= s and e <= y for x, y in spans) and e > s:
            inside.append(n + j - 1)
            if o0 <= s and e <= o1:
                rows.append(n + j - 1)
    return ids, file_pos, thought, inside, rows


def self_copy(text, st):
    m = P["CALL"].search(text or "")
    if not m or m.group(1) != "edit_file":
        return False
    a = dict(P["ARG"].findall(m.group(2)))
    old = a.get("old_string") or ""
    if not old or st["excerpt"].count(old) == 1:
        return False
    bad = [l.strip() for l in old.split("\n") if l.strip() and l.strip() not in st["excerpt"]]
    th = text[: text.find("<|tool_call>")]
    return bool(bad) and all(l in th for l in bad)


def build(tok, samples, ex):
    from huggingface_hub import snapshot_download
    import transformers.models.gemma4.modeling_gemma4 as mg
    from sne.conditional import fit_readout, separation
    from sne.model import layers, load
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False, attn_implementation="sdpa")
    Ls = layers(model)
    H = Ls[0].self_attn.config.num_attention_heads
    D = Ls[LAYERS[0]].self_attn.head_dim
    mask = nope_mask(D)
    CAP = dict(on=False)
    keys = {}
    share = {}
    EDIT = {}
    orig = mg.ALL_ATTENTION_FUNCTIONS["sdpa"]

    def wrapped(module, query, key, value, attention_mask, **kw):
        out = orig(module, query, key, value, attention_mask, **kw)
        L = module.layer_idx
        if CAP["on"] and L in LAYERS:
            if L == LAYERS[0]:
                keys["k"] = key[0][0].float()  # layer 14's shared, normed, rotated keys
            rows, F, T = CAP["rows"], CAP["F"], CAP["T"]
            q = query[0][:, rows].float()  # H x n x D
            lg = q @ key[0][0].float().T * (kw.get("scaling") or 1.0)
            j = torch.arange(lg.shape[-1], device=lg.device)[None, :]
            lg = lg.masked_fill(~(j <= rows[:, None])[None], float("-inf"))
            p = torch.softmax(lg, -1)
            mf, mt = p[:, :, F].sum(-1).mean(-1), p[:, :, T].sum(-1).mean(-1)  # per head
            share[L] = (mf.cpu(), mt.cpu())
        return out
    mg.ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
    st = {}
    hooks = []
    for L in LAYERS:
        hooks.append(Ls[L].self_attn.q_proj.register_forward_pre_hook(lambda m, a, L=L: st.__setitem__(("x", L), a[0][0].float())))
        hooks.append(Ls[L].self_attn.q_proj.register_forward_hook(lambda m, a, o, L=L: st.__setitem__(("q", L), o[0].float())))
    kF, kT, nF, nT = torch.zeros(D), torch.zeros(D), 0, 0
    acts = {L: dict(pos=[], neg=[]) for L in LAYERS}
    rq = {L: [] for L in LAYERS}
    base_share = {L: [] for L in LAYERS}
    rows_all = []
    g = torch.Generator().manual_seed(98)
    rng = random.Random(98)
    with torch.no_grad():
        for k, (s, a) in enumerate(samples):
            prompt = P["G"]["render"](tok, s["prefix"], True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids, F, T, inside, rows = regions(tok, prompt, p_ids, a["text"], s["excerpt"])
            if len(F) < 20 or len(T) < 20 or len(rows) < 3:
                continue
            CAP.update(on=True, rows=torch.tensor(rows, device="cuda"), F=torch.tensor(F, device="cuda"), T=torch.tensor(T, device="cuda"))
            model(input_ids=torch.tensor([p_ids + a_ids], device="cuda"), use_cache=False, logits_to_keep=1)
            CAP["on"] = False
            K = keys["k"]
            kF += K[F].sum(0).cpu()
            kT += K[T].sum(0).cpu()
            nF += len(F)
            nT += len(T)
            for L in LAYERS:
                base_share[L].append(share[L])
                X = st[("x", L)]
                ip = torch.tensor(inside)[torch.randperm(len(inside), generator=g)[:30]]
                neg = T + rng.sample(range(1, len(p_ids)), min(40, len(p_ids) - 1))
                ineg = torch.tensor(neg)[torch.randperm(len(neg), generator=g)[:30]]
                acts[L]["pos"].append(X[ip.cuda()].half().cpu())
                acts[L]["neg"].append(X[ineg.cuda()].half().cpu())
                qr = st[("q", L)][torch.tensor(rows, device="cuda")].view(len(rows), H, D)
                rq[L].append(qr.pow(2).mean(-1).sqrt().cpu())  # n x H
            rows_all.append((s, a, p_ids, a_ids, F, T, rows))
            if k % 20 == 0:
                print(f"[fot] sample {k}/{len(samples)} used {len(rows_all)}", flush=True)
    if len(rows_all) < 10:
        raise RuntimeError(f"only {len(rows_all)} samples had a file region, a thought and copy rows; no direction")
    diff = (kF / max(1, nF) - kT / max(1, nT)) * mask
    gap = float(diff.norm())
    d = diff / diff.norm()
    r = torch.randn(D, generator=g) * mask
    r = r - (r @ d) * d
    rd = r / r.norm()
    rep = dict(samples_used=len(rows_all), file_minus_thought_key_gap_nope=round(gap, 4), layers={})
    vec = {}
    for L in LAYERS:
        n_ = len(acts[L]["pos"])
        cut = int(n_ * 0.8)
        v = fit_readout(torch.cat(acts[L]["pos"][:cut]).float(), torch.cat(acts[L]["neg"][:cut]).float(), lam=1.0)
        sep = separation(v, torch.cat(acts[L]["pos"][cut:]), torch.cat(acts[L]["neg"][cut:]))
        mf = torch.stack([b[0] for b in base_share[L]]).mean(0)
        mt = torch.stack([b[1] for b in base_share[L]]).mean(0)
        gam = Ls[L].self_attn.q_norm.weight.float().cpu()
        rms = torch.cat(rq[L]).median(0).values  # per head
        vec[L] = dict(v=v, gam=gam, rms=rms)
        rep["layers"][L] = dict(readout_check=sep, base_file_mass=[round(float(x), 4) for x in mf], base_thought_mass=[round(float(x), 4) for x in mt])
    ex.log("file-over-thought:", json.dumps(rep))

    def B_for(heads, c, direction):
        out = {}
        for L in LAYERS:
            Bv = torch.zeros(H * D)
            for (LL, h) in heads:
                if LL == L:
                    gm = vec[L]["gam"]
                    Bv[h * D:(h + 1) * D] = c * float(vec[L]["rms"][h]) * torch.where(gm.abs() > 1e-2, direction / gm, torch.zeros_like(gm))
            if Bv.abs().sum() > 0:
                out[L] = Bv
        return out

    def measure(Bs):
        hs = []
        for L, Bv in Bs.items():
            vv, bb = vec[L]["v"].cuda(), Bv.cuda()
            hs.append(Ls[L].self_attn.q_proj.register_forward_hook(lambda m, a, o, vv=vv, bb=bb: o + (a[0].float() @ vv)[..., None].to(o.dtype) * bb.to(o.dtype)))
        res = {L: [] for L in LAYERS}
        with torch.no_grad():
            for (s, a, p_ids, a_ids, F, T, rows) in rows_all[-16:]:
                CAP.update(on=True, rows=torch.tensor(rows, device="cuda"), F=torch.tensor(F, device="cuda"), T=torch.tensor(T, device="cuda"))
                model(input_ids=torch.tensor([p_ids + a_ids], device="cuda"), use_cache=False, logits_to_keep=1)
                CAP["on"] = False
                for L in LAYERS:
                    res[L].append(share[L])
        for h in hs:
            h.remove()
        return {L: (torch.stack([x[0] for x in v]).mean(0), torch.stack([x[1] for x in v]).mean(0)) for L, v in res.items()}
    base_m = measure({})
    arms = {}
    for nm, heads in HEADSETS.items():
        for gp in GAPS:
            c = gp / max(gap, 1e-6)
            Bs = B_for(heads, c, d)
            m = measure(Bs)
            Bc = B_for(heads, c, rd)
            mc = measure(Bc)
            info = {f"L{L}h{h}": dict(file=round(float(m[L][0][h]), 4), thought=round(float(m[L][1][h]), 4),
                                      base_file=round(float(base_m[L][0][h]), 4), base_thought=round(float(base_m[L][1][h]), 4),
                                      control_file=round(float(mc[L][0][h]), 4), control_thought=round(float(mc[L][1][h]), 4)) for L, h in heads}
            arms[f"{nm}_g{gp}"] = dict(heads=heads, c=c, B=Bs, Bc=Bc)
            rep[f"{nm}_g{gp}"] = dict(c=round(c, 4), heads=info)
            print(f"[fot] {nm} gap {gp}: {json.dumps(info)[:900]}", flush=True)
    mg.ALL_ATTENTION_FUNCTIONS["sdpa"] = orig
    for h in hooks:
        h.remove()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return arms, vec, rep


def save_q(path, vec, Bs):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    sd = {}
    for L, Bv in Bs.items():
        base = f"base_model.model.model.language_model.layers.{L}.self_attn.q_proj"
        sd[f"{base}.lora_A.weight"] = vec[L]["v"].view(1, -1).float().contiguous()
        sd[f"{base}.lora_B.weight"] = Bv.view(-1, 1).float().contiguous()
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=1, lora_alpha=1, target_modules=["q_proj"],
                   layers_to_transform=sorted(Bs), lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
                   use_rslora=False), open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    return path


with Experiment("EXP-098", "file-over-thought", category="improvement", depends_on=["EXP-092", "EXP-095"],
                motivation="Half to two thirds of E2B's rewritten old_strings are copied from its own thought rather than the file; a gated query shift in "
                           "the copy heads' position-free dimensions moves their attention from the thought to the file, as an exact rank-1 q_proj LoRA.",
                hypothesis="File-over-thought steering raises applying bench answers on held-out states against the same-server unedited model (exact McNemar "
                           "p < 0.05) and a random-direction control, with no fewer correct answers and no significant MBPP+ loss.",
                config=dict(headsets=HEADSETS, gaps=GAPS, layers=LAYERS, selection="fit split, most applying", test="dev + test (257)",
                            retention="MBPP+ 378", control="random unit direction in the position-free dimensions, orthogonal to d, same heads and size")) as ex:
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    # fail fast: the official vLLM must load a q_proj LoRA on a KV-shared layer before an hour is spent on directions
    from safetensors.torch import save_file
    chk = os.path.join(OUT, "load_check")
    os.makedirs(chk, exist_ok=True)
    save_file({f"base_model.model.model.language_model.layers.{L}.self_attn.q_proj.lora_A.weight": torch.zeros(1, 1536) for L in LAYERS}
              | {f"base_model.model.model.language_model.layers.{L}.self_attn.q_proj.lora_B.weight": torch.zeros(8 * 512, 1) for L in LAYERS},
              os.path.join(chk, "adapter_model.safetensors"))
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=1, lora_alpha=1, target_modules=["q_proj"],
                   layers_to_transform=LAYERS, lora_dropout=0.0, bias="none", fan_in_fan_out=False, init_lora_weights=True, inference_mode=True,
                   use_rslora=False), open(os.path.join(chk, "adapter_config.json"), "w"), indent=2)
    srv = P["Server"]({"chk": chk}, os.path.join(OUT, "vllm_load_check.log"))
    try:
        r = srv._req(dict(model="chk", prompt="def add(a, b):\n    return", max_tokens=6, temperature=0.0))
        ex.log("q_proj LoRA load check on layers", LAYERS, ":", repr(r["choices"][0]["text"]))
    finally:
        srv.close()
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:24]
    states, _ = P["build_states"](tasks, ex)
    rng = random.Random(92)
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
    by_id = {s["iid"]: s for s in states}
    saved = json.load(open(os.path.join(ROOT, "artifacts_local", "EXP-092", "fit_samples.json"), encoding="utf-8"))
    samples = [(by_id[a["iid"]], a) for a in saved if a["iid"] in by_id and by_id[a["iid"]]["split"] == "fit" and "<|tool_call>" in a["text"]]
    random.Random(98).shuffle(samples)
    samples = samples[: (12 if SMOKE else 120)]
    fit = [s for s in states if s["split"] == "fit"]
    held = [s for s in states if s["split"] in ("dev", "test")]
    arms, vec, rep = build(tok, samples, ex)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    ex.save_json("directions.json", rep)
    loras = {k: save_q(os.path.join(OUT, k), vec, a["B"]) for k, a in arms.items()}

    def run(srv, name, ss):
        outs = srv.answers(name, [prompts[s["iid"]] for s in ss])
        return [dict(P["score"](o[0], s), selfcopy=self_copy(o[0], s)) for s, o in zip(ss, outs)]

    def all_arms(srv, names, ss):
        from concurrent.futures import ThreadPoolExecutor as TP
        with TP(len(names)) as tp:
            return dict(zip(names, tp.map(lambda k: run(srv, k, ss), names)))
    srv = P["Server"](loras, os.path.join(OUT, "vllm_scan.log"))
    try:
        scan = all_arms(srv, list(loras) + ["base"], fit)
    finally:
        srv.close()
    counts = {k: dict(applies=sum(r["applies"] for r in v), correct=sum(r["correct"] for r in v), self_copies=sum(r["selfcopy"] for r in v)) for k, v in scan.items()}
    ex.log("fit-split scan (of", len(fit), "):", counts)
    best = max(list(loras), key=lambda k: (counts[k]["applies"], counts[k]["correct"], -float(k.split("_g")[1]), -len(arms[k]["heads"])))
    ctrl = save_q(os.path.join(OUT, "control"), vec, arms[best]["Bc"])
    srv = P["Server"]({"selected": loras[best], "control": ctrl}, os.path.join(OUT, "vllm_test.log"))
    try:
        res = all_arms(srv, ["base", "selected", "control"], held)
        mb = bench.mbppplus("dev") + bench.mbppplus("test")
        if SMOKE:
            mb = mb[:6]
        from sne.evaluate import run_suite
        mbpp = {}
        for name in ("base", "selected", "control"):
            summ, rws = run_suite(Q["Suite"](srv, name), tok, mb, max_tokens=8192)
            mbpp[name] = {r_["task_id"]: r_["passed"] for r_ in rws}
            ex.log(f"[fot] MBPP+ {name}: {summ['passed']}/{summ['n']}")
    finally:
        srv.close()
    col = {k: {m: [r_[m] for r_ in v] for m in ("applies", "correct", "selfcopy")} for k, v in res.items()}
    x, y, p = P["mcnemar"](col["selected"]["applies"], col["base"]["applies"])
    cx, cy, cp = P["mcnemar"](col["selected"]["correct"], col["base"]["correct"])
    tid = [t["task_id"] for t in mb]
    mx, my, mp = P["mcnemar"]([mbpp["selected"][i] for i in tid], [mbpp["base"][i] for i in tid])
    tot = {k: {m: sum(v[m]) for m in ("applies", "correct", "selfcopy")} for k, v in col.items()}
    gate = (p < 0.05 and x > y and tot["selected"]["applies"] > tot["control"]["applies"] and tot["selected"]["correct"] >= tot["base"]["correct"]
            and not (my > mx and mp < 0.05))
    ex.result(directions=rep, fit_scan=counts, selected=best,
              test=dict(n=len(held), totals=tot, applies_mcnemar=dict(selected_only=x, base_only=y, p=round(p, 4)),
                        correct_mcnemar=dict(selected_only=cx, base_only=cy, p=round(cp, 4))),
              mbpp=dict(n=len(tid), base=sum(mbpp["base"].values()), selected=sum(mbpp["selected"].values()), control=sum(mbpp["control"].values()),
                        selected_only=mx, base_only=my, p=round(mp, 4)),
              gate_passed=gate,
              headline=f"file-over-thought {best}: test applies unedited {tot['base']['applies']}/{len(held)}, selected {tot['selected']['applies']} "
                       f"(control {tot['control']['applies']}), McNemar p = {round(p, 4)}; self-copies {tot['base']['selfcopy']} -> {tot['selected']['selfcopy']}; "
                       f"correct {tot['base']['correct']} -> {tot['selected']['correct']}; MBPP+ {sum(mbpp['base'].values())} -> "
                       f"{sum(mbpp['selected'].values())} of {len(tid)}; gate {'passed' if gate else 'not passed'}")

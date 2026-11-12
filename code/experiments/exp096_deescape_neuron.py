"""EXP-096: a de-escaping neuron for E2B's edit_file calls, training-free.

Finding behind it (2026-11-08, the five unedited official runs EXP-044b to 044f, 584 edit_file calls): the harness shows file
contents to the model as JSON (json.dumps), so every double quote in the code appears as backslash-quote, and E2B copies the escape
into its tool call. 78 of the 364 failed edits have backslash-quote in old_string, and 26 of the 220 applied edits wrote a
backslash-quote that was not in old_string into the code (a broken line). On EXP-092's patch bench (the same JSON format) 41 of
E2B's 257 greedy edit answers have backslash-quote in old_string, while no excerpt contains one; unescaping alone would make 23 of the
non-applying ones apply.

The neuron, at layer L: down_proj += u v^T (an exact rank-1 LoRA).
  Read-out v: a ridge read-out on the MLP input that is about 1 at positions that write a string argument of a tool call and about
    0 on thought and prompt positions (bf16, teacher-forced on EXP-092's fit samples of E2B's own answers).
  Write u: raises the logit of each unescaped token against its escaped twin. The vocabulary has 30 tokens containing
    backslash-quote; those without an escaped backslash whose unescaped spelling is also a token form the pairs. The direction is the
    mean, over pairs, of unembedding(unescaped) - unembedding(escaped), carried back through the final norm and the post-feedforward
    norm as in sne.conditional.write_vector, then scaled so that the measured mean margin shift at E2B's own escape positions equals
    the target (up to 4 rescales; reported as infeasible if it stays more than 25% off).

Pre-registered (D-068, before any result):
  Layers: the 2 of {28, 30, 32, 34} with the best held-out read-out ranking (positive above negative, on random pairs).
  Targets: mean margin shift of 6 and 12 at the escape positions. Four neurons (2 layers x 2 targets).
  Selection: on EXP-092's fit split (257 states, unedited greedy known), the neuron with the most applying answers (ties: fewer
    escape errors, then the smaller target, then the later layer).
  Test: EXP-092's dev and test splits together (257 states, untouched by the read-out, the calibration and the selection):
    unedited, the selected neuron, and a control neuron (same read-out, write along a random direction orthogonal to u, same norm).
  Retention: MBPP+ all 378 problems (cap 8,192 tokens), unedited, selected and control.
  Gate for an official agent run (seed 1, D-065): on test, applying answers beat the unedited model (exact McNemar two-sided p < 0.05)
    and the control; correct answers are not fewer than the unedited model's; MBPP+ is not lower than the unedited model's with
    McNemar p < 0.05.
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
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)  # bench, scoring, Server, save_neuron, mcnemar
_src94 = open(os.path.join(HERE, "exp094_bos_value_edit.py"), encoding="utf-8").read()
Q = {"__file__": os.path.join(HERE, "exp094_bos_value_edit.py"), "__name__": "exp094"}
exec(compile(_src94.split('with Experiment("EXP-094"')[0], "exp094", "exec"), Q)  # Suite (MBPP+ through the bench server)
B92 = os.path.join(ROOT, "artifacts_local", "EXP-092")
OUT = os.path.join(ROOT, "artifacts_local", "EXP-096")
CAND = [28, 30, 32, 34]
TARGETS = [6.0, 12.0]
ESC = '\\"'
STR = __import__("re").compile(r'<\|"\|>(.*?)<\|"\|>', __import__("re").S)


def escape_pairs(tok):
    vocab = tok.get_vocab()
    out = []
    for p, i in vocab.items():
        if ESC in p and "\\\\" not in p:
            j = vocab.get(p.replace(ESC, '"'))
            if j is not None:
                out.append((i, j))
    return sorted(out)


def escape_error(text, st):
    """An edit_file answer whose old_string or new_string carries a backslash-quote that the code shown does not contain."""
    m = P["CALL"].search(text or "")
    if not m or m.group(1) != "edit_file" or ESC in st["excerpt"]:
        return False
    a = dict(P["ARG"].findall(m.group(2)))
    return ESC in (a.get("old_string") or "") or ESC in (a.get("new_string") or "")


def positions(tok, prompt_len, text):
    """Answer token ids, the positions that predict a string-argument token of the first tool call, and those of thought tokens."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    tc = text.find("<|tool_call>")
    if tc < 0:
        return ids, [], [], []
    spans = [m.span(1) for m in STR.finditer(text, tc)]
    inside, thought = [], []
    for j, (a, b) in enumerate(offs):
        if b <= tc:
            thought.append(prompt_len + j - 1)
        elif any(s <= a and b <= e for s, e in spans) and b > a:
            inside.append((prompt_len + j - 1, ids[j]))
    return ids, inside, thought, spans


def build(tok, fit_samples, ex):
    from huggingface_hub import snapshot_download
    from sne.conditional import fit_readout, separation
    from sne.model import layers, load, text_model
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False)
    Ls, tm = layers(model), text_model(model)
    W = model.get_output_embeddings().weight
    pairs = escape_pairs(tok)
    esc_of = {i: j for i, j in pairs}
    E = W.float().cpu()
    e = torch.stack([E[j] - E[i] for i, j in pairs]).mean(0)
    wf = tm.norm.weight.float().cpu()
    ex.log("escape pairs", len(pairs), "| |e|", round(float(e.norm()), 4))
    st = {}
    hooks = [tm.norm.register_forward_hook(lambda m, a, o: st.__setitem__("hf", o[0].float())),
             tm.norm.register_forward_pre_hook(lambda m, a: st.__setitem__("hpre", a[0][0].float()))]
    for L in CAND:
        hooks.append(Ls[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: st.__setitem__(("a", L), a[0][0].float())))
        hooks.append(Ls[L].mlp.register_forward_hook(lambda m, a, o, L=L: st.__setitem__(("y", L), o[0].float())))
    g = torch.Generator().manual_seed(96)
    rng = random.Random(96)
    rows, events = [], []
    acts = {L: dict(pos=[], neg=[], rms=[]) for L in CAND}
    rms_h = []
    cands = [(s, a) for s, a in fit_samples if "<|tool_call>" in a["text"]]
    rng.shuffle(cands)
    with_esc = [(s, a) for s, a in cands if ESC in a["text"][a["text"].find("<|tool_call>"):]]
    use = (with_esc[:60] + [x for x in cands if x not in with_esc[:60]])[: (24 if SMOKE else 160)]
    with torch.no_grad():
        for k, (s, a) in enumerate(use):
            prompt = P["G"]["render"](tok, s["prefix"], True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids, inside, thought, _ = positions(tok, len(p_ids), a["text"])
            if not inside:
                continue
            model(input_ids=torch.tensor([p_ids + a_ids], device="cuda"), use_cache=False, logits_to_keep=1)
            pin = torch.tensor([p for p, _ in inside])
            neg = torch.tensor(thought + rng.sample(range(1, len(p_ids)), min(40, len(p_ids) - 1)))
            ip, ineg = pin[torch.randperm(len(pin), generator=g)[:30]], neg[torch.randperm(len(neg), generator=g)[:30]]
            for L in CAND:
                A = st[("a", L)]
                acts[L]["pos"].append(A[ip.cuda()].half().cpu())
                acts[L]["neg"].append(A[ineg.cuda()].half().cpu())
                acts[L]["rms"].append(float(st[("y", L)][pin.cuda()].pow(2).mean(-1).sqrt().median()))
            rms_h.append(float(st["hpre"][pin.cuda()].pow(2).mean(-1).sqrt().median()))
            ev = [(p, t, esc_of[t]) for p, t in inside if t in esc_of]
            rows.append(dict(s=s, a=a, p_ids=p_ids, a_ids=a_ids, ev=ev, inside=[p for p, _ in inside], thought=thought))
            if k % 40 == 0:
                print(f"[deesc] activations {k}/{len(use)}", flush=True)
    for h in hooks[2:]:
        h.remove()
    rep = dict(pairs=len(pairs), samples=len(rows), samples_with_escape=sum(1 for r in rows if r["ev"]),
               escape_events=sum(len(r["ev"]) for r in rows), layers={})
    ranked = []
    for L in CAND:
        n = len(acts[L]["pos"])
        cut = int(n * 0.8)
        v = fit_readout(torch.cat(acts[L]["pos"][:cut]).float(), torch.cat(acts[L]["neg"][:cut]).float(), lam=1.0)
        hp, hn = torch.cat(acts[L]["pos"][cut:]).float() @ v, torch.cat(acts[L]["neg"][cut:]).float() @ v
        i1, i2 = torch.randint(len(hp), (20000,), generator=g), torch.randint(len(hn), (20000,), generator=g)
        auc = float((hp[i1] > hn[i2]).float().mean())
        rep["layers"][L] = dict(readout_rank_acc=round(auc, 4), readout_check=separation(v, torch.cat(acts[L]["pos"][cut:]), torch.cat(acts[L]["neg"][cut:])))
        ranked.append((auc, L, v))
        print(f"[deesc] layer {L} {rep['layers'][L]}", flush=True)
    ranked.sort(key=lambda x: (-x[0], -x[1]))
    keep = ranked[:2]
    rep["selected_layers"] = [L for _, L, _ in keep]
    cal = [r for r in rows if r["ev"]][-24:]
    rep["calibration_samples"] = len(cal)
    if not cal:
        raise RuntimeError("no escape events in the fit samples used; the neuron cannot be calibrated")
    Wf = W.float()

    def margins(r):
        hf_ = st["hf"]
        pos = torch.tensor([p for p, _, _ in r["ev"]], device="cuda")
        z = hf_[pos] @ Wf.T
        z = 30.0 * torch.tanh(z / 30.0)
        i = torch.tensor([t for _, t, _ in r["ev"]], device="cuda")
        j = torch.tensor([u for _, _, u in r["ev"]], device="cuda")
        m = z.gather(1, j[:, None])[:, 0] - z.gather(1, i[:, None])[:, 0]
        still = (z.argmax(-1) == i).float()
        others = torch.tensor(rng.sample(r["inside"], min(20, len(r["inside"]))) + rng.sample(r["thought"], min(20, len(r["thought"]))) if r["thought"]
                              else rng.sample(r["inside"], min(20, len(r["inside"]))), device="cuda")
        top = (hf_[others] @ Wf.T).argmax(-1)
        return m.cpu(), still.cpu(), top.cpu()

    base = []
    with torch.no_grad():
        for r in cal:
            rng.seed(hash(r["s"]["iid"]) % 10 ** 9)
            model(input_ids=torch.tensor([r["p_ids"] + r["a_ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
            base.append(margins(r))
    rep["base_margin_median"] = round(float(torch.cat([b[0] for b in base]).median()), 3)
    rep["base_escape_top1_share"] = round(float(torch.cat([b[1] for b in base]).mean()), 3)

    def measure(L, v, u):
        vv, uu = v.cuda(), u.cuda()
        h = Ls[L].mlp.down_proj.register_forward_hook(lambda m, a, o: o + (a[0].float() @ vv)[..., None].to(o.dtype) * uu.to(o.dtype))
        sh, still, changed = [], [], []
        with torch.no_grad():
            for r, b in zip(cal, base):
                rng.seed(hash(r["s"]["iid"]) % 10 ** 9)
                model(input_ids=torch.tensor([r["p_ids"] + r["a_ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
                m, s_, top = margins(r)
                sh.append(m - b[0])
                still.append(s_)
                changed.append((top != b[2]).float())
        h.remove()
        return float(torch.cat(sh).mean()), float(torch.cat(still).mean()), float(torch.cat(changed).mean())
    out = {}
    for auc, L, v in keep:
        w = Ls[L].post_feedforward_layernorm.weight.float().cpu()
        sc = float(Ls[L].layer_scalar.float().cpu())
        rms_y = float(torch.tensor(acts[L]["rms"]).median())
        u1 = rms_y * float(torch.tensor(rms_h).median()) / sc * (e / e.pow(2).sum()) / (w * wf)  # first order: +1 margin where v.a = 1
        for T in TARGETS:
            c, tries, best = T, [], None
            for _ in range(5):
                got, still, chg = measure(L, v, c * u1)
                tries.append(dict(c=round(c, 3), shift=round(got, 3), escape_still_top1=round(still, 3), other_top1_changed=round(chg, 3)))
                if best is None or got > best[1]:
                    best = (c, got, still, chg)
                if got > 0 and abs(got - T) <= 0.25 * T:
                    break
                c = c * T / got if got > 0.05 * T else c * 2.0
            c, got, still, chg = best
            feasible = abs(got - T) <= 0.25 * T
            u = c * u1
            r_ = torch.randn(u.shape, generator=g)
            r_ = r_ - (r_ @ u) / (u @ u) * u
            uc = r_ / r_.norm() * u.norm()
            gc_, sc_, cc_ = measure(L, v, uc)
            out[f"d_L{L}_t{int(T)}"] = dict(L=L, T=T, u=u, uc=uc, v=v)
            rep["layers"][L][f"target_{int(T)}"] = dict(tries=tries, feasible=feasible, shift=round(got, 3), escape_still_top1=round(still, 3),
                                                       other_top1_changed=round(chg, 3), control=dict(shift=round(gc_, 3), escape_still_top1=round(sc_, 3),
                                                                                                      other_top1_changed=round(cc_, 3)))
            print(f"[deesc] L{L} target {T}: {rep['layers'][L][f'target_{int(T)}']}", flush=True)
    for h in hooks[:2]:
        h.remove()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return out, rep


with Experiment("EXP-096", "deescape-neuron", category="improvement", depends_on=["EXP-092"],
                motivation="E2B copies the JSON escape of the harness's file view into its edits: 78 of 364 failed edits in five unedited official runs carry "
                           "backslash-quote in old_string and 26 of 220 applied edits write one into the code. A neuron that fires inside tool-call strings "
                           "and favours each unescaped token over its escaped twin targets that step without training.",
                hypothesis="The de-escaping neuron raises the share of applying bench answers on held-out states against the unedited model (exact McNemar "
                           "p < 0.05) and against a control neuron, with no fewer correct answers and no significant MBPP+ loss.",
                config=dict(candidate_layers=CAND, targets=TARGETS, selection="fit split, most applying", test="dev + test splits (257)",
                            retention="MBPP+ 378, cap 8,192", control="random write orthogonal to u, same norm, same read-out")) as ex:
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:24]
    states, _ = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    rng = random.Random(92)  # EXP-092's split, reproduced exactly
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
    samples = [(by_id[a["iid"]], a) for a in saved if a["iid"] in by_id and by_id[a["iid"]]["split"] == "fit"]
    fit = [s for s in states if s["split"] == "fit"]
    held = [s for s in states if s["split"] in ("dev", "test")]
    for nm, ss in (("fit", fit), ("dev+test", held)):
        ex.log(f"unedited on {nm}: states {len(ss)}, applies {sum(base[s['iid']]['applies'] for s in ss)}, correct {sum(base[s['iid']]['correct'] for s in ss)}, "
               f"escape errors {sum(escape_error(base[s['iid']].get('text'), s) for s in ss)}")
    VEC = os.path.join(OUT, "vectors.pt")
    if os.path.exists(VEC):  # a relaunch after a crash resumes after the bf16 step
        saved_vec = torch.load(VEC)
        neurons, rep = saved_vec["neurons"], saved_vec["rep"]
        ex.log("resumed: neurons loaded from", VEC)
    else:
        neurons, rep = build(tok, samples, ex)
        torch.save(dict(neurons=neurons, rep=rep), VEC)
    import gc
    gc.collect()
    torch.cuda.empty_cache()  # the HF model must leave the GPU before vLLM starts
    ex.save_json("directions.json", rep)
    loras = {k: P["save_neuron"](os.path.join(OUT, k), d["L"], d["u"], d["v"]) for k, d in neurons.items()}

    def run(srv, name, ss):
        outs = srv.answers(name, [prompts[s["iid"]] for s in ss])
        return [dict(P["score"](o[0], s), esc=escape_error(o[0], s)) for s, o in zip(ss, outs)]

    def all_arms(srv, names, ss):  # arms in parallel, so vLLM's 128 slots stay full through each arm's slow tail
        from concurrent.futures import ThreadPoolExecutor as TP
        with TP(len(names)) as tp:
            return dict(zip(names, tp.map(lambda k: run(srv, k, ss), names)))
    SCAN = os.path.join(OUT, "fit_scan.json")
    if os.path.exists(SCAN):
        scan = json.load(open(SCAN, encoding="utf-8"))
        ex.log("resumed: fit-split scan loaded from", SCAN)
    else:
        srv = P["Server"](loras, os.path.join(OUT, "vllm_scan.log"))
        try:
            scan = all_arms(srv, list(loras) + ["base"], fit)  # D-071: unedited re-run in the same server
        finally:
            srv.close()
        json.dump(scan, open(SCAN, "w", encoding="utf-8"))
    counts = {k: dict(applies=sum(r["applies"] for r in v), correct=sum(r["correct"] for r in v), escape_errors=sum(r["esc"] for r in v)) for k, v in scan.items()}
    counts["base_stored"] = dict(applies=sum(base[s["iid"]]["applies"] for s in fit), correct=sum(base[s["iid"]]["correct"] for s in fit),
                          escape_errors=sum(escape_error(base[s["iid"]].get("text"), s) for s in fit))
    ex.log("fit-split scan (of", len(fit), "):", counts)
    best = max(list(loras), key=lambda k: (counts[k]["applies"], -counts[k]["escape_errors"], -neurons[k]["T"], neurons[k]["L"]))
    d = neurons[best]
    ctrl = P["save_neuron"](os.path.join(OUT, "control"), d["L"], d["uc"], d["v"])
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
            mbpp[name] = {r["task_id"]: r["passed"] for r in rws}
            ex.log(f"[deesc] MBPP+ {name}: {summ['passed']}/{summ['n']}")
    finally:
        srv.close()
    ba = [r["applies"] for r in res["base"]]  # D-071: the unedited arm of the same server run
    bc = [r["correct"] for r in res["base"]]
    be = [r["esc"] for r in res["base"]]
    sa, sc, se = [r["applies"] for r in res["selected"]], [r["correct"] for r in res["selected"]], [r["esc"] for r in res["selected"]]
    ca, cc, ce = [r["applies"] for r in res["control"]], [r["correct"] for r in res["control"]], [r["esc"] for r in res["control"]]
    x, y, p = P["mcnemar"](sa, ba)
    tid = [t["task_id"] for t in mb]
    mx, my, mp = P["mcnemar"]([mbpp["selected"][i] for i in tid], [mbpp["base"][i] for i in tid])
    mbpp_ok = not (my > mx and mp < 0.05)
    gate = p < 0.05 and x > y and sum(sa) > sum(ca) and sum(sc) >= sum(bc) and mbpp_ok
    ex.result(states=len(states), directions=rep, fit_scan=counts, selected=best,
              test=dict(n=len(held), applies=dict(base=sum(ba), selected=sum(sa), control=sum(ca), selected_only=x, base_only=y, p=round(p, 4)),
                        correct=dict(base=sum(bc), selected=sum(sc), control=sum(cc)),
                        escape_errors=dict(base=sum(be), selected=sum(se), control=sum(ce))),
              mbpp=dict(n=len(tid), base=sum(mbpp["base"].values()), selected=sum(mbpp["selected"].values()), control=sum(mbpp["control"].values()),
                        selected_only=mx, base_only=my, p=round(mp, 4)),
              gate_passed=gate,
              headline=f"de-escaping neuron {best}: test applies unedited {sum(ba)}/{len(held)}, selected {sum(sa)} (control {sum(ca)}), McNemar p = {round(p, 4)}; "
                       f"escape errors {sum(be)} -> {sum(se)}; correct {sum(bc)} -> {sum(sc)}; MBPP+ {sum(mbpp['base'].values())} -> "
                       f"{sum(mbpp['selected'].values())} of {len(tid)}; gate {'passed' if gate else 'not passed'}")

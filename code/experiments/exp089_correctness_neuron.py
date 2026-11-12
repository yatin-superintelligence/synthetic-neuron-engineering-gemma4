"""EXP-089: correctness neurons for E2B (idea I3, training-free). A synthetic neuron fires while E2B writes its answer and writes
the direction that separates E2B's own correct solutions from its incorrect ones.

Selection data: EXP-030's corpus. It holds MBPP's original train and validation splits, 317 tasks after every MBPP+ overlap was
removed. For each task it has E2B's greedy answer and two samples, each run against the task's tests. In 88 tasks E2B wrote both a
correct and an incorrect solution. These paired contrasts hold the task fixed, so the direction reflects correctness, not which
tasks are easy.
Literature (survey of 2026-11-08):
  forward steering along a correct-minus-incorrect direction raised MBPP+ pass@1 by up to 15.6% on Qwen2.5-Coder-1.5B-Instruct
    ("Interpreting Code Correctness in Language Models through Activation Steering", ICML 2026);
  an SAE-based study found steering fixed 4% of wrong programs and broke 15% of correct ones (arXiv 2510.02917).
Here the steering is a neuron, so it ships as an exact rank-1 LoRA. It is also conditional: its read-out is about 1 while E2B
writes its answer and about 0 on the prompt.

Pre-registered (D-061, before any result):
  Directions: the residual stream after layer L, averaged over the answer tokens of each solution (at most 512), for L in
    {6, 9, 12, 15, 18, 21, 24, 27}.
    d_L = mean, over the fitting tasks that have both outcomes, of (mean of correct solutions minus mean of incorrect ones).
    Tasks are split 2:1 into fitting and held out (seed 89).
    Two layers are used: those with the best held-out pairwise accuracy (the share of held-out mixed tasks whose correct
    solutions project higher on d_L than their incorrect ones); ties are broken by AUC over all held-out solutions.
  Neuron at layer L: down_proj += u v^T.
    Read-out v: weighted ridge (lambda 1) over the MLP activations entering down_proj, 1 on answer tokens and 0 on prompt tokens.
      Fitted on 150 fitting solutions, at most 48 tokens of each kind per solution; checked on 30 held-out solutions.
    Write u: shifts the residual after layer L by alpha * |delta mu_L| along d_L where the read-out is 1. It passes through the
      post-feedforward norm and the layer scalar to first order. It is then calibrated by a forward pass with the neuron on 10
      held-out solutions, so the measured mean shift along d_L on answer tokens equals the target.
  Scan: MBPP+ dev (189), Q4_0, greedy, no thinking, up to 4,096 new tokens; the unedited model, and alpha in {1, 3, 9} at each of
    the two layers.
  Selection: most dev passes (ties: smaller alpha, then the earlier layer).
  Confirmation: MBPP+ test (189), up to 8,192 tokens, at Q4_0 and F16. Three arms: the unedited model, the selected neuron, and a
    control neuron. The control has the same read-out, and its write is along a random direction orthogonal to d_L, calibrated to
    the same shift size.
  Gate for the agent stage: at Q4_0 the selected neuron must beat the unedited model (exact McNemar, two-sided, p < 0.05) and beat
    the control neuron, and must not be lower than the unedited model at F16.
The bf16 model's activations are gathered in a separate process (phase 1), so the laptop never holds the HF model and a
llama-server at once (the EXP-051 memory fault, 2026-11-08).
"""
import glob
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import MODELS, ROOT, SMOKE  # noqa: E402

LAYERS = [6, 9, 12, 15, 18, 21, 24, 27]
ALPHAS = [1, 3, 9]
MAX_ANS = 512
OUT = os.path.join(ROOT, "scratch", "preflight", "EXP-089") if SMOKE else os.path.join(ROOT, "artifacts_local", "EXP-089")
VEC = os.path.join(OUT, "vectors.pt")
REP = os.path.join(OUT, "phase1.json")


def corpus():
    c = json.load(open(glob.glob(os.path.join(ROOT, "experiments", "EXP-030_*", "corpus.json"))[0], encoding="utf-8"))
    rows = []
    for r in c:
        seen = set()
        for k in ("code_greedy", "code_s1", "code_s2"):
            if r[k] in seen or not r[k].strip():
                continue
            seen.add(r[k])
            rows.append(dict(task=str(r["task_id"]), prompt=r["code_prompt"], answer=r[k], ok=bool(r[f"pass_{k}"])))
    return rows


def groups(rows, idx):
    by = {}
    for i in idx:
        by.setdefault(rows[i]["task"], {True: [], False: []})[rows[i]["ok"]].append(i)
    return by


def phase1():
    from sne.conditional import fit_readout, separation
    from sne.model import device, empty_cache, layers, load
    os.makedirs(OUT, exist_ok=True)
    rows = corpus()
    tasks = sorted({r["task"] for r in rows})
    rng = random.Random(89)
    rng.shuffle(tasks)
    held = set(tasks[: len(tasks) // 3])
    if SMOKE:  # keep a few tasks with both outcomes on each side
        mixed = [t for t, v in groups(rows, range(len(rows))).items() if v[True] and v[False]]
        keep = set(mixed[:8]) | set(tasks[:6])
        held = set(mixed[:3]) | (held & keep)
        rows = [r for r in rows if r["task"] in keep]
    for r in rows:
        r["held"] = r["task"] in held
    model, tok = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    dev = device()
    Ls = layers(model)

    def ids_of(r):
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        a = tok(r["answer"], add_special_tokens=False)["input_ids"][:MAX_ANS]
        return torch.tensor([[tok.bos_token_id] + p + a], device=dev), 1 + len(p)

    st = {}
    hooks = [Ls[L].register_forward_hook(lambda m, a, o, L=L: st.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float()))
             for L in LAYERS]
    R = None
    with torch.no_grad():
        for i, r in enumerate(rows):
            ids, s = ids_of(r)
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
            if R is None:
                R = torch.zeros(len(rows), len(LAYERS), st[LAYERS[0]].shape[-1])
            for j, L in enumerate(LAYERS):
                R[i, j] = st[L][s:].mean(0).cpu()
            if i % 50 == 0:
                print(f"[corr] residual means {i}/{len(rows)}", flush=True)
                empty_cache()
    for h in hooks:
        h.remove()
    fit_i = [i for i, r in enumerate(rows) if not r["held"]]
    held_i = [i for i, r in enumerate(rows) if r["held"]]
    rep = dict(n_solutions=len(rows), n_correct=sum(r["ok"] for r in rows), n_tasks=len({r["task"] for r in rows}), layers={})
    dirs = {}
    for j, L in enumerate(LAYERS):
        diffs = [R[v[True], j].mean(0) - R[v[False], j].mean(0) for v in groups(rows, fit_i).values() if v[True] and v[False]]
        d = torch.stack(diffs).mean(0)
        dn = d / d.norm()
        wins = n = 0
        for v in groups(rows, held_i).values():
            if v[True] and v[False]:
                n += 1
                wins += float((R[v[True], j] @ dn).mean() > (R[v[False], j] @ dn).mean())
        sp = R[[i for i in held_i if rows[i]["ok"]], j] @ dn
        sn = R[[i for i in held_i if not rows[i]["ok"]], j] @ dn
        auc = float((sp[:, None] > sn[None, :]).float().mean()) if len(sp) and len(sn) else 0.5
        rep["layers"][L] = dict(fit_mixed_tasks=len(diffs), held_mixed_tasks=n, pairwise_acc=round(wins / max(1, n), 4), auc=round(auc, 4),
                                dmu_norm=round(float(d.norm()), 4), residual_norm=round(float(R[:, j].norm(dim=-1).mean()), 2))
        dirs[L] = (dn, float(d.norm()))
        print(f"[corr] layer {L}: {rep['layers'][L]}", flush=True)
    ranked = sorted(LAYERS, key=lambda L: (-rep["layers"][L]["pairwise_acc"], -rep["layers"][L]["auc"]))
    sel = ranked[:4]  # D-061a: read-out data for the 4 best; the first 2 where the neuron can deliver the alpha = 1 shift are used
    rep["ranked_layers"] = ranked
    # read-outs, MLP output scale and calibrated writes at the candidate layers
    acts = {L: dict(pos=[], neg=[], cpos=[], cneg=[], rms=[]) for L in sel}
    hk = []
    for L in sel:
        hk.append(Ls[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: st.__setitem__(("a", L), a[0][0].float())))
        hk.append(Ls[L].mlp.register_forward_hook(lambda m, a, o, L=L: st.__setitem__(("y", L), o[0].float())))
    g = torch.Generator().manual_seed(89)
    rng.shuffle(fit_i)
    rng.shuffle(held_i)
    sample, check = fit_i[:150], held_i[:30]
    with torch.no_grad():
        for k, i in enumerate(sample + check):
            ids, s = ids_of(rows[i])
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
            for L in sel:
                a = st[("a", L)].cpu()
                ans, pr = a[s:], a[1:s]
                pa = ans[torch.randperm(len(ans), generator=g)[:48]]
                pn = pr[torch.randperm(len(pr), generator=g)[:48]]
                tgt = acts[L]
                (tgt["pos"] if k < len(sample) else tgt["cpos"]).append(pa.half())
                (tgt["neg"] if k < len(sample) else tgt["cneg"]).append(pn.half())
                tgt["rms"].append(float(st[("y", L)][s:].pow(2).mean(-1).sqrt().median()))
            if k % 50 == 0:
                print(f"[corr] read-out data {k}/{len(sample) + len(check)}", flush=True)
                empty_cache()
    for h in hk:
        h.remove()
    vec = {}
    for L in sel:
        if len(vec) == 2:
            break
        A = acts[L]
        Xp, Xn = torch.cat(A["pos"]).float(), torch.cat(A["neg"]).float()
        v = fit_readout(Xp, Xn, lam=1.0)
        sep = separation(v, torch.cat(A["cpos"]).float(), torch.cat(A["cneg"]).float())
        dn, dmu = dirs[L]
        w = Ls[L].post_feedforward_layernorm.weight.float().cpu()
        sc = float(Ls[L].layer_scalar.float().cpu())
        rms_y = float(torch.tensor(A["rms"]).median())
        r = torch.randn(dn.shape, generator=g)
        r = r - (r @ dn) * dn
        rn = r / r.norm()

        def u_for(direction):
            return dmu * direction * rms_y / (sc * w)

        def measured(u, direction):
            """Mean shift of the residual after layer L along `direction` on answer tokens, with the neuron added."""
            vv, uu = v.to(dev), u.to(dev)
            res = {}
            h1 = Ls[L].register_forward_hook(lambda m, a, o: res.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
            shifts, cos = [], []
            with torch.no_grad():
                for i in check[:10]:
                    ids, s = ids_of(rows[i])
                    model(input_ids=ids, use_cache=False, logits_to_keep=1)
                    h0 = res["h"][s:]
                    h2 = Ls[L].mlp.down_proj.register_forward_hook(lambda m, a, o: o + (a[0].float() @ vv)[..., None].to(o.dtype) * uu.to(o.dtype))
                    model(input_ids=ids, use_cache=False, logits_to_keep=1)
                    h2.remove()
                    dh = (res["h"][s:] - h0)
                    shifts.append(float((dh @ direction).mean()))
                    cos.append(float(torch.nn.functional.cosine_similarity(dh.mean(0), direction, dim=0)))
            h1.remove()
            return sum(shifts) / len(shifts), sum(cos) / len(cos)

        def calibrate(direction, target):
            """D-061a: the post-feedforward RMSNorm caps and bends what an MLP write can do (preflight, 2026-11-08: at layer 6 the
            first-order write moved the residual the wrong way). Rescale up to 4 times; a write whose shift has the wrong sign is halved.
            Feasible: within 25% of the target and cosine with the direction at least 0.9."""
            u = u_for(direction) * (target / dmu)
            hist = []
            m, c = 0.0, 0.0
            for _ in range(4):
                m, c = measured(u, direction)
                hist.append((round(m, 3), round(c, 3)))
                if abs(m - target) <= 0.05 * target:
                    break
                u = u * (target / m) if m > 1e-9 else u * 0.5
            return u, dict(achieved=round(m, 4), target=round(target, 4), cosine=round(c, 3), steps=hist,
                           feasible=bool(abs(m - target) <= 0.25 * target and c >= 0.9))

        u1, f1 = calibrate(dn, dmu)
        rep["layers"][L].update(readout_check=sep, rms_y=round(rms_y, 4), layer_scalar=round(sc, 4), calib={1: f1})
        if not f1["feasible"]:
            rep["layers"][L]["used"] = False
            print(f"[corr] layer {L} not used: alpha 1 {f1} | kept so far {list(vec)}", flush=True)
            continue
        us, ucs, feas = {1: u1}, {}, {1: True}
        for a in ALPHAS:
            if a != 1:
                us[a], fa = calibrate(dn, a * dmu)
                rep["layers"][L]["calib"][a] = fa
                feas[a] = fa["feasible"]
            ucs[a], fc = calibrate(rn, a * dmu)
            rep["layers"][L].setdefault("control_calib", {})[a] = fc
        vec[L] = dict(v=v, u=us, uc=ucs, feasible=feas, d=dn, dmu=dmu)
        rep["layers"][L]["used"] = True
        print(f"[corr] layer {L} neuron: {rep['layers'][L]}", flush=True)
    rep["selected_layers"] = list(vec)
    torch.save(vec, VEC)
    json.dump(rep, open(REP, "w"), indent=1)


def neuron(L, u, v):
    def fn(e, bw, prec):
        e.add(L, "mlp.down_proj", u.view(-1, 1), v.view(1, -1))
        e.meta.update(layer=L, rank=1)
    return fn


def outcomes(ex, arm, suite):
    return {r["task_id"]: r["passed"] for r in map(json.loads, open(os.path.join(ex.dir, f"outputs_{arm}_{suite}.jsonl"), encoding="utf-8"))}


if __name__ == "__main__" and "--phase1" in sys.argv:
    phase1()
    sys.exit(0)

from sne.artifacts import build_adapters  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402
from sne.stats import paired_compare  # noqa: E402

with Experiment("EXP-089", "correctness-neuron", category="improvement", depends_on=["EXP-030"],
                motivation="E2B's agent failures are mostly wrong patches; a training-free neuron that moves its code towards its own correct solutions "
                           "targets that bottleneck directly (idea I3; activation-steering results of 2026).",
                hypothesis="A correctness neuron at a selected layer and strength raises MBPP+ test passes against the unedited model (exact McNemar "
                           "p < 0.05) and against a control neuron of matched strength at Q4_0, and does not lower them at F16.",
                config=dict(layers=LAYERS, alphas=ALPHAS, max_answer_tokens=MAX_ANS, readout="ridge 1 on answer tokens, 0 on prompt tokens",
                            scan="MBPP+ dev 189, Q4_0, cap 4,096", confirm="MBPP+ test 189, Q4_0 and F16, cap 8,192",
                            control="same read-out, write along a random direction orthogonal to d_L, same calibrated shift")) as ex:
    os.makedirs(OUT, exist_ok=True)
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--phase1"], check=True)
    rep = json.load(open(REP, encoding="utf-8"))
    vec = torch.load(VEC)
    ex.save_json("phase1.json", rep)
    ex.log("phase 1", json.dumps({k: v for k, v in rep.items() if k != "layers"}), "| layers:", json.dumps(rep["layers"])[:3000])
    sel = rep["selected_layers"]
    if not sel:
        ex.result(headline="no candidate layer can deliver the alpha = 1 shift through its MLP (post-feedforward norm); nothing to evaluate", phase1=rep)
        sys.exit(0)
    scan = {}
    for L in sel:
        for a in ALPHAS:
            if not vec[L]["feasible"][a]:
                ex.log(f"layer {L} alpha {a}: the neuron cannot deliver this shift (D-061a); arm not run", rep["layers"][str(L)]["calib"].get(str(a)))
                continue
            scan[f"corr_L{L}_a{a}"] = build_adapters(ex, f"corr_L{L}_a{a}", neuron(L, vec[L]["u"][a], vec[L]["v"]), precisions=("q4_0",))
    arms = [Arm("base_q4_0", "q4_0")] + [Arm(f"{n}_q4_0", "q4_0", lora=scan[n]["q4_0"]) for n in scan]
    res, comps = evaluate_arms(ex, arms, ["mbpp_dev"], reference="base", max_tokens=4096)
    dev_table = {a.name: res[a.name]["mbpp_dev"]["passed"] if "passed" in res[a.name]["mbpp_dev"] else res[a.name]["mbpp_dev"]["pass_rate"]
                 for a in arms}
    ex.result(scan_dev=dev_table, scan_paired=comps)
    best = max(scan, key=lambda n: (dev_table[f"{n}_q4_0"], -int(n.split("_a")[1]), -sel.index(int(n.split("_L")[1].split("_")[0]))))
    L, a = int(best.split("_L")[1].split("_")[0]), int(best.split("_a")[1])
    ex.log("selected", best, "dev", dev_table[f"{best}_q4_0"], "vs unedited", dev_table["base_q4_0"])
    full = build_adapters(ex, "selected", neuron(L, vec[L]["u"][a], vec[L]["v"]))
    ctrl = build_adapters(ex, "control", neuron(L, vec[L]["uc"][a], vec[L]["v"]))
    arms2 = []
    for prec in ("q4_0", "f16"):
        arms2 += [Arm(f"base_{prec}_test", prec), Arm(f"selected_{prec}_test", prec, lora=full[prec]), Arm(f"control_{prec}_test", prec, lora=ctrl[prec])]
    res2, comps2 = evaluate_arms(ex, arms2, ["mbpp_test"], reference="base", max_tokens=8192)
    test_table = {x.name: res2[x.name]["mbpp_test"] for x in arms2}
    stats = {}
    for prec in ("q4_0", "f16"):
        b, s_, c = (outcomes(ex, f"{n}_{prec}_test", "mbpp_test") for n in ("base", "selected", "control"))
        stats[prec] = dict(selected_vs_base=paired_compare(b, s_), selected_vs_control=paired_compare(c, s_), control_vs_base=paired_compare(b, c))
    q, f = stats["q4_0"], stats["f16"]
    gate = (q["selected_vs_base"]["mcnemar_p"] < 0.05 and q["selected_vs_base"]["delta"] > 0 and q["selected_vs_control"]["delta"] > 0
            and f["selected_vs_base"]["delta"] >= 0)
    ex.result(selected=best, test={k: {kk: v.get(kk) for kk in ("passed", "pass_rate", "n")} for k, v in test_table.items()}, stats=stats, gate_passed=gate,
              headline=f"selected {best}: MBPP+ test Q4_0 {test_table['selected_q4_0_test'].get('pass_rate')} vs unedited "
                       f"{test_table['base_q4_0_test'].get('pass_rate')} (control {test_table['control_q4_0_test'].get('pass_rate')}); "
                       f"F16 {test_table['selected_f16_test'].get('pass_rate')} vs {test_table['base_f16_test'].get('pass_rate')}; gate {'passed' if gate else 'not passed'}")

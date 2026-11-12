"""EXP-097: escape-blind attention values, training-free, aimed at E2B copying JSON escapes into its edits.

Background: the harness shows file contents as JSON, so a double quote in the code reaches E2B as backslash-quote, and E2B copies the
escape into old_string or new_string (78 of 364 failed edits and 26 of 220 applied edits in five unedited official runs; 25 of 257
unedited bench answers on dev and test). EXP-096 tried to fix the output side, a neuron raising each unescaped token over its escaped
twin, and its writes could not move the margin (layer 28: 0.23 of a 6.0 target, below its random control): the post-feedforward norm
caps what an added MLP output can do in Gemma 4. This edit works on the input side instead: what the attention heads read.

The edit: at each chosen KV-computing layer L (0 to 14; layers 15 to 34 reuse the values of layers 13 and 14), the escape feature e_L
is the mean, over aligned pairs, of the normed v_proj input at an escaped quote minus that at the same quote shown plain. Each pair is
one double quote of a real excerpt, once inside the JSON tool output the agent sees (escaped) and once in a plain code block. With
e^ = e_L / |e_L|: W_v' = W_v - alpha (W_v e^) e^T, an exact rank-1 v_proj LoRA. A head that copies from an escaped quote then reads a
value without (alpha = 1) or with half (alpha = 0.5) the escape component. EXP-094b's control showed a rank-1 v_proj projection along a
random direction at all 15 layers leaves MBPP+ unchanged at F16 (9 of 12, unedited 9 of 12).

Pre-registered (D-073, before any result):
  Arms: layer set {13, 14} or all of 0 to 14, x alpha in {0.5, 1}: four edits.
  Selection on EXP-092's fit split (257 states): the most applying answers (ties: fewer escape errors, smaller alpha, fewer layers);
    the unedited arm runs in the same server (D-071).
  Test on EXP-092's dev and test splits (257 states): unedited, selected, and a control edit (same layers and alpha, a random unit
    direction per layer, seed 97), all in one server.
  Retention: MBPP+ all 378 problems (cap 8,192 tokens), unedited, selected and control.
  Gate for an official agent run (seed 1): test applies above the unedited arm (exact McNemar two-sided p < 0.05) and above the
    control; correct answers not fewer; MBPP+ not lower with McNemar p < 0.05. Escape errors are reported for every arm.
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
exec(compile(_src94.split('with Experiment("EXP-094"')[0], "exp094", "exec"), Q)  # Suite, save_v_adapter
B92 = os.path.join(ROOT, "artifacts_local", "EXP-092")
OUT = os.path.join(ROOT, "artifacts_local", "EXP-097")
N_COMPUTE = 15
LAYER_SETS = {"L13-14": [13, 14], "L0-14": list(range(N_COMPUTE))}
ALPHAS = [0.5, 1.0]
ESC = '\\"'


def escape_error(text, st):
    m = P["CALL"].search(text or "")
    if not m or m.group(1) != "edit_file" or ESC in st["excerpt"]:
        return False
    a = dict(P["ARG"].findall(m.group(2)))
    return ESC in (a.get("old_string") or "") or ESC in (a.get("new_string") or "")


def quote_positions(tok, text, start, raw, escaped):
    """Token positions (in tok(text)) of each double quote of raw, as rendered from char offset start (escaped or plain)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    out, c = [], start
    for ch in raw:
        if ch == '"':
            at = c + (1 if escaped else 0)  # the quote character itself (after the backslash when escaped)
            j = next((k for k, (a, b) in enumerate(offs) if a <= at < b), None)
            out.append(j)
        step = len(json.dumps(ch)[1:-1]) if escaped else 1
        c += step
    return enc["input_ids"], out


def directions(tok, states, ex):
    """The escape feature per KV-computing layer from aligned (escaped, plain) quotes of real excerpts."""
    from huggingface_hub import snapshot_download
    from sne.model import layers, load
    hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/hf_qat")
    model, _ = load(hf, ple_on_cpu=False)
    Ls = layers(model)
    st = {}
    hooks = [Ls[i].self_attn.v_proj.register_forward_pre_hook(lambda m, a, i=i: st.__setitem__(i, a[0][0].float())) for i in range(N_COMPUTE)]
    diffs = {i: [] for i in range(N_COMPUTE)}
    n_pairs = 0
    rng = random.Random(97)
    use = [s for s in states if s["excerpt"].count('"') >= 4]
    rng.shuffle(use)
    with torch.no_grad():
        for k, s in enumerate(use[: (8 if SMOKE else 80)]):
            raw = s["excerpt"][:6000]
            body = json.dumps(dict(status="ok", filepath="x.py", content=raw))
            pre_e = "<|turn>user\nHere is the tool output:\n"
            text_e = pre_e + body + "<turn|>\n"
            start_e = len(pre_e) + body.index('"content": "') + len('"content": "')
            pre_q = "<|turn>user\nHere is the code:\n```python\n"
            text_q = pre_q + raw + "\n```<turn|>\n"
            ids_e, pe = quote_positions(tok, text_e, start_e, raw, True)
            ids_q, pq = quote_positions(tok, text_q, len(pre_q), raw, False)
            pairs = [(a, b) for a, b in zip(pe, pq) if a is not None and b is not None]
            if not pairs:
                continue
            bos = [tok.bos_token_id]
            model(input_ids=torch.tensor([bos + ids_e], device="cuda"), use_cache=False, logits_to_keep=1)
            xe = {i: st[i][[a + 1 for a, _ in pairs]].cpu() for i in range(N_COMPUTE)}
            model(input_ids=torch.tensor([bos + ids_q], device="cuda"), use_cache=False, logits_to_keep=1)
            for i in range(N_COMPUTE):
                xq = st[i][[b + 1 for _, b in pairs]].cpu()
                diffs[i].append(xe[i] - xq)
            n_pairs += len(pairs)
            if k % 20 == 0:
                print(f"[escblind] excerpt {k} pairs so far {n_pairs}", flush=True)
    for h in hooks:
        h.remove()
    g = torch.Generator().manual_seed(97)
    vec, rep = {}, dict(pairs=n_pairs, layers={})
    for i in range(N_COMPUTE):
        D = torch.cat(diffs[i])
        n = len(D)
        cut = max(1, int(n * 0.8))
        e = D[:cut].mean(0)
        eh = e / e.norm()
        held = D[cut:]
        frac = float(((held @ eh).pow(2).sum() / held.pow(2).sum().clamp_min(1e-12))) if len(held) else None
        cons = float((torch.nn.functional.normalize(held, dim=-1) @ eh).mean()) if len(held) else None
        W = Ls[i].self_attn.v_proj.weight.float().cpu()
        r = torch.randn(eh.shape, generator=g)
        rh = r / r.norm()
        vec[i] = dict(A=eh, B=-(W @ eh), Ac=rh, Bc=-(W @ rh))
        rep["layers"][i] = dict(escape_norm=round(float(e.norm()), 4), heldout_share_along=round(frac, 4) if frac is not None else None,
                                heldout_mean_cos=round(cons, 4) if cons is not None else None)
    ex.log("escape directions", json.dumps(rep))
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return vec, rep


def save(path, vec, layers_, alpha, ctrl):
    sub = {i: dict(A=(vec[i]["Ac"] if ctrl else vec[i]["A"]), B=alpha * (vec[i]["Bc"] if ctrl else vec[i]["B"])) for i in layers_}
    return Q["save_v_adapter"](path, sub, "bos")  # save_v_adapter reads A and B under the "bos" key names


with Experiment("EXP-097", "escape-blind-values", category="improvement", depends_on=["EXP-092", "EXP-096", "EXP-094b"],
                motivation="E2B copies the JSON escape of the harness's file view into its edits; output-side writes are capped by Gemma 4's norms (EXP-096), "
                           "so this edit removes the escape feature from what the attention heads read, as an exact rank-1 v_proj LoRA.",
                hypothesis="Escape-blind values raise the share of applying bench answers on held-out states against the unedited model in the same server "
                           "(exact McNemar p < 0.05) and against a random-direction control, with no fewer correct answers and no significant MBPP+ loss.",
                config=dict(layer_sets=LAYER_SETS, alphas=ALPHAS, selection="fit split, most applying", test="dev + test splits (257)",
                            retention="MBPP+ 378, cap 8,192", control="random unit direction per layer, seed 97, same layers and alpha")) as ex:
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tasks, _ = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:24]
    states, _ = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
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
    fit = [s for s in states if s["split"] == "fit"]
    held = [s for s in states if s["split"] in ("dev", "test")]
    vec, rep = directions(tok, fit, ex)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    ex.save_json("directions.json", rep)
    loras = {f"{nm}_a{a}": save(os.path.join(OUT, f"{nm}_a{a}"), vec, ls, a, False) for nm, ls in LAYER_SETS.items() for a in ALPHAS}

    def run(srv, name, ss):
        outs = srv.answers(name, [prompts[s["iid"]] for s in ss])
        return [dict(P["score"](o[0], s), esc=escape_error(o[0], s)) for s, o in zip(ss, outs)]

    def all_arms(srv, names, ss):
        from concurrent.futures import ThreadPoolExecutor as TP
        with TP(len(names)) as tp:
            return dict(zip(names, tp.map(lambda k: run(srv, k, ss), names)))
    srv = P["Server"](loras, os.path.join(OUT, "vllm_scan.log"))
    try:
        scan = all_arms(srv, list(loras) + ["base"], fit)
    finally:
        srv.close()
    counts = {k: dict(applies=sum(r["applies"] for r in v), correct=sum(r["correct"] for r in v), escape_errors=sum(r["esc"] for r in v)) for k, v in scan.items()}
    ex.log("fit-split scan (of", len(fit), "):", counts)
    best = max(list(loras), key=lambda k: (counts[k]["applies"], -counts[k]["escape_errors"], -float(k.split("_a")[1]), -len(LAYER_SETS[k.split("_a")[0]])))
    nm, a = best.split("_a")[0], float(best.split("_a")[1])
    ctrl = save(os.path.join(OUT, "control"), vec, LAYER_SETS[nm], a, True)
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
            ex.log(f"[escblind] MBPP+ {name}: {summ['passed']}/{summ['n']}")
    finally:
        srv.close()
    col = {k: dict(applies=[r["applies"] for r in v], correct=[r["correct"] for r in v], esc=[r["esc"] for r in v]) for k, v in res.items()}
    x, y, p = P["mcnemar"](col["selected"]["applies"], col["base"]["applies"])
    tid = [t["task_id"] for t in mb]
    mx, my, mp = P["mcnemar"]([mbpp["selected"][i] for i in tid], [mbpp["base"][i] for i in tid])
    tot = {k: {m: sum(v[m]) for m in ("applies", "correct", "esc")} for k, v in col.items()}
    gate = (p < 0.05 and x > y and tot["selected"]["applies"] > tot["control"]["applies"] and tot["selected"]["correct"] >= tot["base"]["correct"]
            and not (my > mx and mp < 0.05))
    ex.result(directions=rep, fit_scan=counts, selected=best, test=dict(n=len(held), totals=tot, applies_mcnemar=dict(selected_only=x, base_only=y, p=round(p, 4))),
              mbpp=dict(n=len(tid), base=sum(mbpp["base"].values()), selected=sum(mbpp["selected"].values()), control=sum(mbpp["control"].values()),
                        selected_only=mx, base_only=my, p=round(mp, 4)),
              gate_passed=gate,
              headline=f"escape-blind values {best}: test applies unedited {tot['base']['applies']}/{len(held)}, selected {tot['selected']['applies']} "
                       f"(control {tot['control']['applies']}), McNemar p = {round(p, 4)}; escape errors {tot['base']['esc']} -> {tot['selected']['esc']}; "
                       f"correct {tot['base']['correct']} -> {tot['selected']['correct']}; MBPP+ {sum(mbpp['base'].values())} -> "
                       f"{sum(mbpp['selected'].values())} of {len(tid)}; gate {'passed' if gate else 'not passed'}")

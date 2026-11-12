"""EXP-080: synthetic decision neurons for E2B, distilled in closed form from Gemma 4 31B's choices at E2B's own states (EXP-079).

Each neuron is a new rank-1 unit in E2B's last MLP (layer 34), as in EXP-076: a ridge read-out v over the 12,288 activations entering
down_proj, about 1 where 31B would take action Y at a state where E2B took action X and about 0 elsewhere, and a write u that raises
the first token of Y over the first token of X by c logits where the read-out is 1. D' = D + u v^T, an exact rank-1 LoRA. No gradient
training: v is a ridge solution, u follows from the unembedding (src/sne/conditional.py).

Fixed before the run:
- States and labels: artifacts_local/EXP-079/teacher_labels.jsonl. A state is a positive for the pair (X, Y) when E2B took X and at
  least 3 of 31B's 4 samples took Y. Pairs are tool-name changes only (X and Y both tools); changes to or from stopping need another
  position and write, and are only counted.
- A pair gets a neuron when it has at least 25 positives in the dev split. Negatives: every other dev state.
- Activations: the bf16 reference E2B, at the tool-name position (E2B's own thinking, then <channel|><|tool_call>call:), with the
  same context as the teacher saw (EXP-079's cut to 22,000 tokens). EXP-076 used only the last 3,000 tokens; the EXP-076 N_reread
  read-out is re-scored here at full context, to see whether its separation survives.
- Write strength, set from data (changed 2026-11-08 00:35 UTC, before this script ran, after EXP-081): for each neuron, the smallest
  c in 3, 4, ..., 10 for which the first-order prediction gap + c * read-out > 0 holds at 90% of its dev positives (gap = logit of
  Y's first token minus X's), else 10. EXP-081 showed why: with c = 3 the EXP-076 re-read neuron could turn only about 7 more points
  of its moments, because the read-out is about 0.6 at full context and the model's own preferences vary widely. Reported per neuron:
  the predicted share of test positives where Y beats X, before and after; and the predicted harm, the share of test states where E2B
  took X and 31B agreed with X that would now prefer Y. Ridge 1, as in EXP-076.
- N_reread is refitted here at full context (same label as EXP-076: the previous tool call was an edit_file that failed; X edit_file,
  Y read_file), with the same rules, so EXP-076's neuron can be retested in its strong form.
- Check: separation on the test split, which the fit never sees; a control per neuron (read-out fitted to as many random dev states,
  same write and strength).
- Outputs: one adapter per neuron, one with all neurons (rank k) and its control, in artifacts_local/EXP-080; counts in the record.
Precision: built on bf16; the delta does not depend on base weights, so the adapter is exact at every precision. Real-task tests follow.
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, ROOT, SMOKE  # noqa: E402
from sne.conditional import fit_readout, separation, write_vector  # noqa: E402
from sne.model import device, empty_cache, layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import tokenizer  # noqa: E402
import exp079_teacher_decisions as T  # noqa: E402  (state list and prompt cut, shared with the teacher run)

LAYER, C, LAM, MIN_POS = 34, 3.0, 1.0, 25
LABELS = os.path.join(ROOT, "artifacts_local", "EXP-079", "teacher_labels.jsonl")
OUT = os.path.join(ROOT, "artifacts_local", "EXP-080")
TOOLS = [d[0] for d in T.L.TOOL_DECLS] + ["code_analyzer_agent"]


def first_token(tok, name):
    a = tok("<|tool_call>call:", add_special_tokens=False)["input_ids"]
    b = tok("<|tool_call>call:" + name, add_special_tokens=False)["input_ids"]
    assert b[:len(a)] == a, name
    return b[len(a)]


@torch.no_grad()
def gather(model, tok, rows, dev, keep_ids):
    Ls = layers(model)
    st = {}
    h1 = Ls[LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0, -1].float().cpu()))
    h2 = Ls[LAYER].mlp.register_forward_hook(lambda m, a, o: st.__setitem__("y", o[0, -1].float()))
    h3 = Ls[LAYER].register_forward_hook(lambda m, a, o: st.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
    cache = {}
    try:
        for k, r in enumerate(rows):
            t = cache.setdefault(r["trace"], json.load(open(r["trace"], encoding="utf-8")))
            ids, _ = T.prompt(tok, t, r["step"])
            msg = t["steps"][r["step"]].get("message") or ""
            ids = ids + tok(msg + "<channel|><|tool_call>call:", add_special_tokens=False)["input_ids"]
            out = model(input_ids=torch.tensor([[tok.bos_token_id] + ids], device=dev), use_cache=False, logits_to_keep=1)
            r.update(a=st["a"].half(), rms_y=float(st["y"].pow(2).mean().sqrt()), rms_h=float(st["h"].pow(2).mean().sqrt()),
                     logits={i: float(out.logits[0, -1, i]) for i in keep_ids}, n_tokens=len(ids) + 1)
            del out
            if k % 50 == 0:
                empty_cache()
                print(f"[gather] {k}/{len(rows)}", flush=True)
    finally:
        h1.remove(), h2.remove(), h3.remove()


def save_adapter(name, parts):
    from safetensors.torch import save_file
    B = torch.stack([u for u, v in parts], 1).float()
    A = torch.stack([v for u, v in parts], 0).float()
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    base = f"base_model.model.model.language_model.layers.{LAYER}.mlp.down_proj"
    save_file({f"{base}.lora_A.weight": A.contiguous(), f"{base}.lora_B.weight": B.contiguous()}, os.path.join(d, "adapter_model.safetensors"))
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=len(parts), lora_alpha=len(parts),
               target_modules=["down_proj"], layers_to_transform=[LAYER], lora_dropout=0.0, bias="none", fan_in_fan_out=False,
               init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)


def stack(rs):
    return torch.stack([r["a"].float() for r in rs])


def failed(obs):
    t = obs.get("content") if isinstance(obs, dict) else json.dumps(obs or "")
    return isinstance(t, str) and '"status": "error"' in t.replace("'", '"')


with Experiment("EXP-080", "teacher-neurons", category="improvement", depends_on=["EXP-079", "EXP-076"],
                motivation="Add neurons that move E2B towards what Gemma 4 31B does at the same agent state, built in closed form from EXP-079's labels.",
                hypothesis="Pairs (E2B action X, 31B action Y) with enough confident disagreements give read-outs that separate their states on held-out tasks far above control.",
                config=dict(layer=LAYER, logit_shift="calibrated, 3 to 10", ridge=LAM, min_dev_positives=MIN_POS, teacher_min_votes=3, max_prompt=T.MAX_PROMPT)) as ex:
    labels = {(r["run"], r["iid"], r["step"]): r for r in map(json.loads, open(LABELS, encoding="utf-8"))}
    rows = [r for r in T.states() if (r["run"], r["iid"], r["step"]) in labels and r["e2b"] != "STOP"]
    for r in rows:
        lab = labels[(r["run"], r["iid"], r["step"])]
        maj, k = collections.Counter(lab["teacher"]).most_common(1)[0]
        r.update(teacher=lab["teacher"], t_major=maj, t_votes=k)
    if SMOKE:
        rows = rows[:16]
    pairs = collections.Counter((r["e2b"], r["t_major"]) for r in rows if r["t_votes"] >= 3 and r["t_major"] != r["e2b"])
    ex.log("states with a tool call", len(rows), "| confident tool-to-other disagreements", sum(pairs.values()), "| pairs", pairs.most_common(12))
    stop_pairs = collections.Counter((r["e2b"], r["t_major"]) for r in rows if r["t_votes"] >= 3 and r["t_major"] in ("STOP", "UNDECIDED"))
    ex.log("31B would stop or not decide instead (not built):", stop_pairs.most_common(6))
    tok = tokenizer()
    tid = {n: first_token(tok, n) for n in TOOLS}
    ex.log("first tokens", tid)
    from sne.model import load
    model, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    dev = device()
    gather(model, tok, rows, dev, sorted(set(tid.values())))
    fit = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    rng = random.Random(80)
    cache = {}
    for r in rows:  # EXP-076's re-read label: the previous tool call in the same trace was an edit_file whose reply has status "error"
        t = cache.setdefault(r["trace"], json.load(open(r["trace"], encoding="utf-8")))
        prev = [s for j, s in T.L.agent_steps(t) if j < r["step"] and s.get("tool_calls")]
        r["reread"] = bool(prev) and prev[-1]["tool_calls"][0].get("function_name") == "edit_file" and failed(prev[-1].get("observation"))
    out, parts, control = {}, [], []

    def gaps(rs, x, y):
        return np.array([float(r["logits"][tid[y]] - r["logits"][tid[x]]) for r in rs])

    def build(name, x, y, lab):
        pos = [r for r in fit if lab(r)]
        if len(pos) < MIN_POS:
            ex.log(name, "skipped:", len(pos), "dev positives")
            return
        neg = [r for r in fit if not lab(r)]
        v = fit_readout(stack(pos), stack(neg), lam=LAM)
        g, ro = gaps(pos, x, y), stack(pos).matmul(v).numpy()
        c = next((k for k in range(3, 11) if np.mean(g + k * ro > 0) >= 0.9), 10)
        tp, tn = [r for r in test if lab(r)], [r for r in test if not lab(r)]
        keep = [r for r in tn if r["e2b"] == x and r.get("t_major") == x]  # E2B took X and 31B agreed: turning these is harm
        pred = {}
        for key, rs in (("test_positives", tp), ("test_agreeing_x_states", keep)):
            if rs:
                gg, rr_ = gaps(rs, x, y), stack(rs).matmul(v).numpy()
                pred[key] = dict(n=len(rs), y_beats_x_before=round(float(np.mean(gg > 0)), 3), after=round(float(np.mean(gg + c * rr_ > 0)), 3))
        random_states = set(rng.sample(range(len(fit)), len(pos)))
        vp = fit_readout(stack([fit[i] for i in sorted(random_states)]), stack([fit[i] for i in range(len(fit)) if i not in random_states]), lam=LAM)
        u = write_vector(model, LAYER, tid[y], tid[x], float(c), float(np.median([r["rms_y"] for r in pos])), float(np.median([r["rms_h"] for r in pos])))
        out[name] = dict(x=x, y=y, c=c, n_pos=len(pos), n_neg=len(neg), sep_fit=separation(v, stack(pos), stack(neg)),
                         sep_test=separation(v, stack(tp), stack(tn)) if tp and tn else None,
                         control_on_test_positives=round(float(stack(tp).matmul(vp).mean()), 4) if tp else None,
                         dev_gap_median=round(float(np.median(g)), 3), predicted=pred)
        ex.log(name, out[name])
        save_adapter(f"neuron_{name}", [(u, v)])
        save_adapter(f"control_{name}", [(u, vp)])
        torch.save(dict(u=u, v=v, vp=vp, c=c, x=x, y=y), os.path.join(OUT, f"neuron_{name}.pt"))
        if name != "reread_full":
            parts.append((u, v))
            control.append((u, vp))

    for (x, y), n in pairs.most_common():
        if x in tid and y in tid:
            build(f"{x}_to_{y}", x, y, lambda r, x=x, y=y: r["e2b"] == x and r["t_major"] == y and r["t_votes"] >= 3)
    build("reread_full", "edit_file", "read_file", lambda r: r["reread"])
    if parts:
        save_adapter("teacher_all", parts)
        save_adapter("control_all", control)
    # the EXP-076 read-out (fitted at 3,000 tokens) scored at full context, for comparison with EXP-081
    rr = os.path.join(ROOT, "artifacts_local", "EXP-076", "neuron_reread.pt")
    reread = None
    if os.path.exists(rr):
        v76 = torch.load(rr)["v"].float()
        reread = {}
        for sp, rs in (("dev", fit), ("test", test)):
            p, n = [r for r in rs if r["reread"]], [r for r in rs if not r["reread"]]
            reread[sp] = separation(v76, stack(p), stack(n)) if p and n else None
        ex.log("EXP-076 N_reread read-out (fitted at 3,000 tokens) at full context:", reread)
    ex.result(neurons=out, reread_at_full_context=reread, headline=f"{len(out)} teacher neurons built; " + "; ".join(
        f"{k} (c={v['c']}): test positives {v['sep_test']['pos_mean'] if v['sep_test'] else 'n/a'} vs negatives p99 {v['sep_test']['neg_p99'] if v['sep_test'] else 'n/a'}, control {v['control_on_test_positives']}, predicted {v['predicted']}"
        for k, v in out.items()))

"""EXP-127 (D-154, D-154a): two synthetic neurons for Gemma 4 31B, built in closed form from its own official runs, aimed at its runaway runs.

Why (D-153, tools/analysis/runaway_31b.py): on the 40 tasks the reference fix resolves, unedited 31B (EXP-074c, seed 1) solves 25.
Runs that stay normal solve 24 of 28; runaway runs (a response at the 4,096-token thinking cap, or 60+ tool calls) solve 1 of 12,
and 12 of the 15 unsolved tasks are runaway. Where 31B reaches the right lines it solves 25 of 28: its headroom is behaviour.

Neurons (each a new rank-1 unit in the last MLP: a read-out v over the existing neuron activations entering down_proj, about 1 at
its state and about 0 elsewhere, and a write u that shifts next-token logits there; D' = D + u v^T, an exact LoRA):
  N_conclude. Where: sentence ends inside a thought, after the thought's 1,500th token, in thoughts that ran to the cap (at least
  4,000 tokens). Hard negatives: sentence ends in thoughts that closed on their own, and before token 1,500 of capped thoughts; plus
  sampled other thought tokens. Write: raise the end-of-thought token over the tokens most probable there.
  N_commit. Where: the token before the tool name, when the agent has made 30 or more tool calls since its last applied edit (or since
  the start). Negatives: tool-name positions with fewer than 15 such calls. Write: raise "edit" over "read" and "run".
One forward pass per run (D-154a): every state is a prefix of its run's whole conversation and attention is causal, so the
activations at a state's position in the whole run's pass equal those of a pass over the state alone, with the full context the agent
had (no truncation). Each state's prefix is checked token for token against the whole run's tokens; a state that does not match is
dropped and counted. 58 passes replace about 900.
Fit on the dev half's tasks (split seed 20261106), separation reported on the test half's tasks, which are not used to fit; ridge
strength chosen per neuron as the largest held-out positive mean with held-out negatives' 99th percentile under 0.1 (EXP-051's
rule); a neuron that fails the rule is not built. Write strength c per neuron: the median base logit gap at the fitting positives
plus 1, clipped to [3, 9] (9.5 is the single-write cap measured on E2B, D-136). Control: the same writes with random read-outs of
the same norms (seed 127). Adapters (artifacts_local/EXP-127): neurons_both, control_both, neuron_commit, neuron_conclude.
Precision: built on the bf16 QAT checkpoint; the delta does not depend on base weights, so it is exact on the official W4A16 model.
Runs on a Colab G4 with --official. EXP127_MODEL overrides the model (a local folder for the CPU pipeline test).
"""
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import ROOT, SMOKE, conditional as C, loops as L  # noqa: E402
from sne.model import layers  # noqa: E402
from sne.registry import Experiment  # noqa: E402

REPO = os.environ.get("EXP127_MODEL", "google/gemma-4-31B-it-qat-q4_0-unquantized")
TRACES = os.path.join(ROOT, "artifacts_local", "colab", "exp-074c", "results", "traces")
OUT = os.path.join(ROOT, "artifacts_local", "EXP-127")
NAMES = dict(read=1399, edit=8982, run=4503)
CAP_TOKENS, ONSET, STREAK_POS, STREAK_NEG = 4000, 1500, 30, 15
MAX_HARD, MAX_RANDOM, RANDOM_PER_THOUGHT = 8000, 6000, 20
LAMBDAS = (0.01, 0.1, 1.0, 10.0)


def failed(obs):
    t = obs.get("content") if isinstance(obs, dict) else json.dumps(obs or "")
    return isinstance(t, str) and '"status": "error"' in t.replace("'", '"')


def load_traces():
    return {os.path.basename(f)[6:-5]: json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(TRACES, "trace_*.json")))}


def locate(tok, traces, split, rng):
    """Per run: the whole conversation's tokens and, at checked positions, the commit states and the thought sentence ends."""
    close = tok.convert_tokens_to_ids("<channel|>")
    runs, drop = [], collections.Counter()
    for iid, t in traces.items():
        steps = L.agent_steps(t)
        full = tok(L.render(tok, dict(messages=L.messages_before(t, len(t["steps"])))), add_special_tokens=False)["input_ids"]
        commit, ends, streak = [], [], 0
        call_ids = tok("<|tool_call>call:", add_special_tokens=False)["input_ids"]
        for i, s in steps:
            calls = s.get("tool_calls") or []
            if not calls:
                continue
            name = calls[0].get("function_name")
            prev = streak  # calls since the last applied edit, before this one
            label = prev >= STREAK_POS if (prev >= STREAK_POS or prev < STREAK_NEG) else None
            streak = 0 if (name == "edit_file" and not failed(s.get("observation"))) else streak + 1
            # the step's generation prompt must be an exact token prefix of the whole run; the thought runs from there to the
            # end-of-thought token, which the tool call follows (the template puts a newline before that token)
            gen = tok(L.render(tok, dict(messages=L.messages_before(t, i))), add_special_tokens=False)["input_ids"]
            if full[:len(gen)] != gen:
                drop["generation prompt not a prefix"] += 1
                continue
            try:
                stop = full.index(close, len(gen))
            except ValueError:
                drop["thought without end token"] += 1
                continue
            cpos = stop + len(call_ids)  # the last token before the tool name
            if full[stop + 1:cpos + 1] == call_ids and tok.decode(full[cpos + 1:cpos + 6]).startswith(name):
                commit.append(dict(pos=cpos, streak=prev, label=label))
            else:
                drop["tool call not where expected"] += 1
            n = (s.get("metrics") or {}).get("completion_tokens") or 0
            capped = n >= CAP_TOKENS
            span = range(len(gen), stop)
            se = [q for q in span if tok.decode([full[q]]).rstrip(" ").endswith((".", "\n", ":"))]
            for q in se:
                ends.append(dict(pos=q, kind="pos" if capped and q - len(gen) > ONSET else "hard"))
            sset = set(se)
            others = [q for q in span if q not in sset]
            for q in rng.sample(others, min(RANDOM_PER_THOUGHT, len(others))):
                ends.append(dict(pos=q, kind="random"))
        runs.append(dict(iid=iid, split=split.get(iid, "dev"), ids=full, commit=commit, ends=ends))
    return runs, drop, close


@torch.no_grad()
def gather(model, tok, runs, dev, layer):
    """One forward pass per run; activations entering down_proj and the MLP and layer output RMS at the located positions, logits only
    where a write is calibrated (commit states, conclude positives). Returns the summed next-token distribution at conclude positives."""
    Ls = layers(model)
    st = {}
    top_acc = None
    close = tok.convert_tokens_to_ids("<channel|>")
    h1 = Ls[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a: st.__setitem__("a", a[0][0, st["p"]].float().cpu()))
    h2 = Ls[layer].mlp.register_forward_hook(lambda m, a, o: st.__setitem__("y", o[0, st["p"]].float().pow(2).mean(-1).sqrt().cpu()))
    h3 = Ls[layer].register_forward_hook(lambda m, a, o: st.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, st["p"]].float().pow(2).mean(-1).sqrt().cpu()))
    try:
        for k, r in enumerate(runs):
            rows = r["commit"] + r["ends"]
            if not rows:
                continue
            pos = sorted({x["pos"] + 1 for x in rows})  # +1: the bos token is prepended
            lpos = sorted({x["pos"] + 1 for x in rows if "streak" in x or x["kind"] == "pos"})
            st["p"] = torch.tensor(pos, device=dev)
            ids = torch.tensor([[tok.bos_token_id] + r["ids"]], device=dev)
            out = model(input_ids=ids, use_cache=False, logits_to_keep=torch.tensor(lpos or [len(r["ids"])], device=dev))
            lg = out.logits[0].float().cpu()
            at = {p: j for j, p in enumerate(pos)}
            lat = {p: j for j, p in enumerate(lpos)}
            for x in rows:
                j = at[x["pos"] + 1]
                x.update(a=st["a"][j].half(), rms_y=float(st["y"][j]), rms_h=float(st["h"][j]))
                if "streak" in x:
                    q = lg[lat[x["pos"] + 1]]
                    x["gap"] = float(torch.stack([q[NAMES["read"]], q[NAMES["run"]]]).mean() - q[NAMES["edit"]])
                    x["top"] = max(NAMES, key=lambda n: float(q[NAMES[n]]))
                elif x["kind"] == "pos":
                    q = lg[lat[x["pos"] + 1]]
                    sm = torch.softmax(q, -1)
                    top_acc = sm if top_acc is None else top_acc + sm
                    x["gap"] = float(q.max() - q[close])
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[gather] run {k + 1}/{len(runs)}: {len(r['ids'])} tokens, {len(pos)} positions", flush=True)
    finally:
        h1.remove(), h2.remove(), h3.remove()
    return top_acc


def choose(pos, neg, pos_held, neg_held):
    grid = []
    for lam in LAMBDAS:
        v = C.fit_readout(pos, neg, lam=lam)
        grid.append(dict(lam=lam, held=C.separation(v, pos_held, neg_held) if len(pos_held) and len(neg_held) else None))
    ok = [g for g in grid if g["held"] and g["held"]["neg_p99"] < 0.1]
    best = max(ok, key=lambda g: g["held"]["pos_mean"]) if ok else None
    return grid, best


def save_adapter(name, layer, parts, base_name):
    from safetensors.torch import save_file
    B = torch.stack([u for u, v in parts], 1).float()
    A = torch.stack([v for u, v in parts], 0).float()
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    key = f"base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
    save_file({f"{key}.lora_A.weight": A.contiguous(), f"{key}.lora_B.weight": B.contiguous()}, os.path.join(d, "adapter_model.safetensors"))
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=base_name, r=len(parts), lora_alpha=len(parts),
               target_modules=["down_proj"], layers_to_transform=[layer], lora_dropout=0.0, bias="none", fan_in_fan_out=False,
               init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    return d


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sne.chunked_attention import install
    install()  # 31B's head_dim-512 full-attention layers would otherwise build the whole score matrix
    if os.path.isdir(REPO):
        path = REPO
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO, local_dir="/content/hf_31b")
    tok = AutoTokenizer.from_pretrained(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=dev, low_cpu_mem_usage=True).eval()
    for nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
        if hasattr(model.model, nm) and getattr(model.model, nm) is not None:
            setattr(model.model, nm, None)
    return model, tok, dev


def stack(rows):
    return torch.stack([r["a"].float() for r in rows]) if rows else torch.zeros(0, 1)


with Experiment("EXP-127", "31b-runaway-neurons", category="improvement", depends_on=["EXP-074c", "EXP-051", "EXP-076"],
                motivation="31B's headroom on the reference-resolvable tasks is runaway behaviour (D-153): deliberating to the thinking cap "
                           "and exploring without committing. Two closed-form neurons from 31B's own runs act at those states only.",
                hypothesis="Each read-out separates its state on held-out tasks (99th percentile of held-out negatives under 0.1); the agent "
                           "test (EXP-128) then measures runaway runs and tasks solved.",
                config=dict(model=REPO, cap_tokens=CAP_TOKENS, onset=ONSET, streak_pos=STREAK_POS, streak_neg=STREAK_NEG, lambdas=LAMBDAS,
                            max_hard=MAX_HARD, max_random=MAX_RANDOM, one_pass_per_run=True, traces="EXP-074c")) as ex:
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(127)
    split = L.split_map()
    traces = load_traces()
    if SMOKE:
        traces = dict(sorted(traces.items(), key=lambda kv: len(kv[1]["steps"]))[:3])
    model, tok, dev = load_model()
    layer = len(layers(model)) - 1
    base_name = REPO.replace("-qat-q4_0-unquantized", "") if not os.path.isdir(REPO) else "google/gemma-4-31B-it"
    runs, drop, close_id = locate(tok, traces, split, rng)
    commit_all = [x for r in runs for x in r["commit"] if x["label"] is not None]
    ends_all = [x for r in runs for x in r["ends"]]
    ex.log("runs", len(runs), "| tokens per run: median", int(np.median([len(r["ids"]) for r in runs])), "max", max(len(r["ids"]) for r in runs),
           "| commit states", len(commit_all), "positives", sum(x["label"] for x in commit_all),
           "| thought positions: positives", sum(x["kind"] == "pos" for x in ends_all), "hard", sum(x["kind"] == "hard" for x in ends_all),
           "random", sum(x["kind"] == "random" for x in ends_all), "| dropped", dict(drop))
    # cap the ridge size: keep every positive, sample the negatives
    for r in runs:
        r["commit"] = [x for x in r["commit"] if x["label"] is not None]
    hard = [x for x in ends_all if x["kind"] == "hard"]
    rand = [x for x in ends_all if x["kind"] == "random"]
    keep = set(map(id, rng.sample(hard, min(MAX_HARD, len(hard))) + rng.sample(rand, min(MAX_RANDOM, len(rand)))))
    for r in runs:
        r["ends"] = [x for x in r["ends"] if x["kind"] == "pos" or id(x) in keep]
    ex.log("model", REPO, "| layers", layer + 1, "| neurons entering down_proj", layers(model)[layer].mlp.down_proj.in_features)
    top_acc = gather(model, tok, runs, dev, layer)
    for r in runs:
        for x in r["commit"] + r["ends"]:
            x["split"] = r["split"]
            x["iid"] = r["iid"]
    out, parts, ctrl = {}, {}, {}
    torch.manual_seed(127)

    # N_conclude
    ends = [x for r in runs for x in r["ends"]]
    P = [x for x in ends if x["kind"] == "pos"]
    N = [x for x in ends if x["kind"] != "pos"]
    Hd = [x for x in ends if x["kind"] == "hard"]
    fitP, fitN = [x for x in P if x["split"] == "dev"], [x for x in N if x["split"] == "dev"]
    heldP, heldH = [x for x in P if x["split"] == "test"], [x for x in Hd if x["split"] == "test"]
    grid, best = choose(stack(fitP), stack(fitN), stack(heldP), stack(heldH)) if fitP and fitN else ([], None)
    gap = float(np.median([x["gap"] for x in fitP])) if fitP else 0.0
    cc = float(np.clip(gap + 1, 3, 9))
    out["conclude"] = dict(positives=len(P), positive_tasks=len({x["iid"] for x in P}), negatives=len(N), hard=len(Hd), held_positives=len(heldP),
                           grid=grid, chosen=best, median_gap_top_to_close=round(gap, 3), c=cc)
    ex.log("conclude", {k: v for k, v in out["conclude"].items() if k != "grid"})
    if (best or SMOKE) and fitP:
        v = C.fit_readout(stack(P), stack(N), lam=best["lam"] if best else LAMBDAS[-1])
        top = top_acc.topk(50).indices.tolist()
        u = C.write_vector(model, layer, close_id, top, cc, float(np.mean([x["rms_y"] for x in P])), float(np.mean([x["rms_h"] for x in P])))
        rv = torch.randn_like(v)
        parts["conclude"], ctrl["conclude"] = (u, v), (u, rv / rv.norm() * v.norm())

    # N_commit
    com = [x for r in runs for x in r["commit"]]
    fit = [x for x in com if x["split"] == "dev"]
    test = [x for x in com if x["split"] == "test"]
    grid, best = choose(stack([x for x in fit if x["label"]]), stack([x for x in fit if not x["label"]]),
                        stack([x for x in test if x["label"]]), stack([x for x in test if not x["label"]])) if any(x["label"] for x in fit) else ([], None)
    pos_fit = [x for x in fit if x["label"]]
    gap = float(np.median([x["gap"] for x in pos_fit])) if pos_fit else 0.0
    cm = float(np.clip(gap + 1, 3, 9))
    out["commit"] = dict(positives=sum(x["label"] for x in com), positive_tasks=sorted({x["iid"] for x in com if x["label"]}),
                         held_positives=sum(x["label"] for x in test), negatives=sum(not x["label"] for x in com), grid=grid, chosen=best,
                         median_gap_read_run_over_edit=round(gap, 3), c=cm,
                         top_choice_at_positives=dict(collections.Counter(x["top"] for x in com if x["label"])))
    ex.log("commit", {k: v for k, v in out["commit"].items() if k not in ("grid", "positive_tasks")})
    if (best or SMOKE) and pos_fit:
        v = C.fit_readout(stack([x for x in com if x["label"]]), stack([x for x in com if not x["label"]]), lam=best["lam"] if best else LAMBDAS[-1])
        u = C.write_vector(model, layer, NAMES["edit"], [NAMES["read"], NAMES["run"]], cm, float(np.median([x["rms_y"] for x in pos_fit])),
                           float(np.median([x["rms_h"] for x in pos_fit])))
        rv = torch.randn_like(v)
        parts["commit"], ctrl["commit"] = (u, v), (u, rv / rv.norm() * v.norm())

    saved = {}
    for k in parts:
        saved[f"neuron_{k}"] = save_adapter(f"neuron_{k}", layer, [parts[k]], base_name)
    if parts:
        saved["neurons_both"] = save_adapter("neurons_both", layer, list(parts.values()), base_name)
        saved["control_both"] = save_adapter("control_both", layer, list(ctrl.values()), base_name)
    ex.save_json("fit.json", dict(neurons=out, dropped=dict(drop)))
    ex.result(neurons=out, adapters=sorted(saved), layer=layer, dropped=dict(drop),
              headline="; ".join(f"{k}: held-out positives mean {o['chosen']['held']['pos_mean'] if o.get('chosen') else 'n/a'}, "
                                 f"held-out negatives p99 {o['chosen']['held']['neg_p99'] if o.get('chosen') else 'n/a'}, c {o['c']}"
                                 for k, o in out.items()) + f"; adapters {sorted(saved)}")

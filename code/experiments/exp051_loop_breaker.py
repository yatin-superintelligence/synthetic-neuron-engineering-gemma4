"""EXP-051: a closed-form rank-1 loop breaker: close the thought once the model starts repeating itself (LoopBench dev split).

Read-out: weighted ridge on the 12288 activations of the last MLP, fitted on sentence ends after loop onset against sentence
ends in normal thinking and other tokens. Write: raises the end-of-thought token over the tokens most probable at those
points. Regularisation is chosen on a held-out third of dev tasks: the largest mean held-out read-out with a 99th percentile
on held-out normal sentence ends under 0.1. Arms at F16 and Q4_0: base, c in (8, 16, 32), and a control with the strongest
write and a random read-out of equal norm. Screens stop a response once it counts as a loop, since its label cannot change.
The measure is the loop rate on loop contexts, with tool-call rate on both context kinds as the guard against damage.
Context length drives loop risk, so the short-prompt filter used for edits does not apply here.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import MODELS, SMOKE, conditional as C, loops as L  # noqa: E402
from sne.artifacts import build_adapters  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.model import empty_cache, load  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

LAYER = 34
GAINS = (8.0, 16.0, 32.0)
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
SEEDS = (0,)

with Experiment("EXP-051", "loop-breaker-neuron", category="improvement", depends_on=["EXP-040"],
                motivation="Loops take 31% of tokens in EXP-037 and precede 10 of 15 context errors. A sampling setting cannot "
                           "act on repetition of whole sentences, and a read-out of existing neurons can act only where it fires.",
                hypothesis="The last layer already represents repetition well enough for a linear read-out, so a rank-1 edit "
                           "shortens or ends loops without changing normal thinking. The random read-out does nothing.",
                config=dict(layer=LAYER, gains=GAINS, lambdas=LAMBDAS, seeds=SEEDS, ctx_tokens=3000, resp_tokens=1500, split="dev",
                            selection_rule="max held-out pos_mean with hard-negative p99 < 0.1", early_exit=True)) as ex:
    tok = tokenizer()
    cases = L.load("dev")
    tasks = sorted({c["instance_id"] for c in cases})
    random.Random(51).shuffle(tasks)
    held = set(tasks[: max(1, len(tasks) // 3)])
    model, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    d = C.gather_loops(model, tok, cases, layer=LAYER)
    ex.log("positives", len(d["pos"]), "negatives", len(d["neg"]), "hard", len(d["hard"]), "mean top-to-close logit gap", sum(d["gap"]) / len(d["gap"]))
    pm = torch.tensor([t in held for t in d["pos_task"]])
    nm = torch.tensor([t in held for t in d["neg_task"]])
    hm = torch.tensor([t in held for t in d["hard_task"]])
    grid = []
    for lam in LAMBDAS:
        v = C.fit_readout(d["pos"][~pm], d["neg"][~nm], lam=lam)
        grid.append(dict(lam=lam, all_neg=C.separation(v, d["pos"][pm], d["neg"][nm]), hard=C.separation(v, d["pos"][pm], d["hard"][hm])))
        ex.log(grid[-1])
    ok = [g for g in grid if g["hard"]["neg_p99"] < 0.1]
    if not ok and not SMOKE:
        # a read-out that also fires on normal sentence ends would close normal thinking, so replays would only measure damage
        ex.result(grid=grid, headline="read-out failed the selection rule: normal sentence ends score like loop ones; no replays run")
        sys.exit(0)
    best = max(ok or grid, key=lambda g: g["hard"]["pos_mean"])
    v = C.fit_readout(d["pos"], d["neg"], lam=best["lam"])
    rms_y, rms_h = sum(d["rms_y"]) / len(d["rms_y"]), sum(d["rms_h"]) / len(d["rms_h"])
    us = {c: C.write_vector(model, LAYER, d["close_id"], d["top_ids"], c, rms_y, rms_h) for c in GAINS}
    torch.manual_seed(51)
    r = torch.randn_like(v)
    r = r / r.norm() * v.norm()
    ex.save_json("fit.json", dict(grid=grid, chosen=best, rms_y=rms_y, rms_h=rms_h, gap_mean=sum(d["gap"]) / len(d["gap"]), v_norm=float(v.norm())))
    del model, d
    import gc
    gc.collect()
    empty_cache()  # the Intel GPU uses shared system RAM: without this the model kept 8.6 GB while llama-server ran (2026-11-08)

    def edit_of(u, read):
        def fn(e, bw, prec):
            e.add(LAYER, "mlp.down_proj", u.view(-1, 1), read.view(1, -1))
            e.meta.update(layer=LAYER, rank=1)
        return fn
    defs = {f"break_c{int(c)}": (us[c], v) for c in GAINS}
    defs["control_random_read"] = (us[GAINS[-1]], r)
    adapters = {n: build_adapters(ex, n, edit_of(*x)) for n, x in defs.items()}
    out = {}
    names = list(defs)
    for prec in ["q4_0", "f16"]:
        out[prec] = {}
        with LlamaServer(GGUF[prec], loras=[adapters[n][prec] for n in names], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            for name in ["base"] + names:
                scales = {names.index(name): 1.0} if name != "base" else {}
                summ, rows, _ = L.run(srv, tok, cases, lora_scales=scales, seeds=SEEDS, early_exit=True)
                out[prec][name] = summ
                ex.save_json(f"rows_{prec}_{name}.json", rows)
                ex.log(prec, name, summ)
                ex.result(arms=out)
    f = out["q4_0"]
    top = min(names[:-1], key=lambda n: f[n]["loop_rate_on_loop_contexts"][0])
    ex.result(arms=out, chosen=best, headline=f"Q4_0 loop rate {f['base']['loop_rate_on_loop_contexts'][0]:.2f} base, {f[top]['loop_rate_on_loop_contexts'][0]:.2f} {top}, "
                                              f"control {f['control_random_read']['loop_rate_on_loop_contexts'][0]:.2f}; control tool calls {f['base']['tool_call_rate_on_controls'][0]:.2f} to {f[top]['tool_call_rate_on_controls'][0]:.2f}")

"""EXP-042: a closed-form rank-1 edit that makes the model write old_string before new_string (EditBench dev split).

Read-out: a weighted ridge regression on the 12288 neuron activations of the last MLP, fitted on dev tasks, marks the one
decision token after the file path of an edit_file call. Write: the first-order vector that raises the logit of `old` over
`new` by c. The regularisation is chosen on a held-out third of the dev tasks by a fixed rule: the largest mean read-out on
held-out decision points with a 99.9th percentile on held-out other tokens under 0.1. It is then refitted on all dev tasks.
Arms at F16 and Q4_0: base, c in (8, 16, 32), and a control with the strongest write and a random read-out of equal norm.
Evaluated from the recorded thought with a free choice of the next argument, so the edit is judged on what the model does.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import MODELS, SMOKE, conditional as C, editbench as E, loops as L  # noqa: E402
from sne.artifacts import build_adapters  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.lora import apply_torch  # noqa: E402
from sne.model import load  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

LAYER = 34
GAINS = (8.0, 16.0, 32.0)
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
SEEDS = (0, 1)

with Experiment("EXP-042", "order-neuron", category="improvement", depends_on=["EXP-041"],
                motivation="EXP-041 tests whether argument order causes failed edits. If it does, the fix is a change at one "
                           "token, which a system prompt may not reach reliably and sampling settings cannot express.",
                hypothesis="A rank-1 edit on the last MLP flips the first argument to old_string on most edit calls, leaves "
                           "other tokens unchanged, raises the apply rate, and the random read-out does nothing.",
                config=dict(layer=LAYER, gains=GAINS, lambdas=LAMBDAS, seeds=SEEDS, ctx_tokens=3000, split="dev", selection_rule="max held-out pos_mean with neg_p999 < 0.1")) as ex:
    tok = tokenizer()
    cases = E.short_cases(tok, E.load("dev", per_run_task=4), 9000)
    if not cases:
        raise RuntimeError("no states to test")
    tasks = sorted({c["instance_id"] for c in cases})
    random.Random(42).shuffle(tasks)
    held = set(tasks[: max(1, len(tasks) // 3)])
    model, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    d = C.gather(model, tok, cases, layer=LAYER)
    ex.log("positives", len(d["pos"]), "negatives", len(d["neg"]), "baseline old-new logit gap mean", sum(d["gap"]) / len(d["gap"]))
    pm = torch.tensor([t in held for t in d["pos_task"]])
    nm = torch.tensor([t in held for t in d["neg_task"]])
    grid = []
    for lam in LAMBDAS:
        v = C.fit_readout(d["pos"][~pm], d["neg"][~nm], lam=lam)
        s = C.separation(v, d["pos"][pm], d["neg"][nm])
        grid.append(dict(lam=lam, **s))
        ex.log(grid[-1])
    ok = [g for g in grid if g["neg_p999"] < 0.1] or grid
    best = max(ok, key=lambda g: g["pos_mean"])
    v = C.fit_readout(d["pos"], d["neg"], lam=best["lam"])
    rms_y, rms_h = sum(d["rms_y"]) / len(d["rms_y"]), sum(d["rms_h"]) / len(d["rms_h"])
    us = {c: C.write_vector(model, LAYER, d["old_id"], d["new_id"], c, rms_y, rms_h) for c in GAINS}
    torch.manual_seed(42)
    r = torch.randn_like(v)
    r = r / r.norm() * v.norm()
    ex.save_json("fit.json", dict(grid=grid, chosen=best, rms_y=rms_y, rms_h=rms_h, gap_mean=sum(d["gap"]) / len(d["gap"]), v_norm=float(v.norm()), u_norm={str(k): float(x.norm()) for k, x in us.items()}))

    def edit_of(u, read):
        def fn(e, bw, prec):
            e.add(LAYER, "mlp.down_proj", u.view(-1, 1), read.view(1, -1))
            e.meta.update(layer=LAYER, rank=1)
        return fn
    defs = {f"order_c{int(c)}": (us[c], v) for c in GAINS}
    defs["control_random_read"] = (us[GAINS[-1]], r)
    # torch check of the compiled delta: logit gap on real decision points before and after the strongest edit
    from sne.lora import Edit
    e = Edit("check")
    edit_of(us[GAINS[-1]], v)(e, None, None)
    apply_torch(model, e)
    after = C.gather(model, tok, cases[:6], layer=LAYER)
    apply_torch(model, e, sign=-1.0)
    ex.log("gap before", d["gap"][:6], "after strongest edit", after["gap"][:6])
    del model, d
    adapters = {n: build_adapters(ex, n, edit_of(*x)) for n, x in defs.items()}
    out = {}
    names = list(defs)
    for prec in ["f16", "q4_0"]:
        out[prec] = {}
        with LlamaServer(GGUF[prec], loras=[adapters[n][prec] for n in names], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            for name in ["base"] + names:
                scales = {names.index(name): 1.0} if name != "base" else {}
                rows, summ = E.free_key(srv, tok, cases, lora_scales=scales, seeds=SEEDS)
                out[prec][name] = summ
                ex.save_json(f"rows_{prec}_{name}.json", rows)
                ex.log(prec, name, summ)
                ex.result(arms=out)
    f = out["q4_0"]
    top = max(names[:-1], key=lambda n: f[n]["applied"])
    ex.result(arms=out, chosen=best, headline=f"Q4_0 applied {f['base']['applied']:.2f} base, {f[top]['applied']:.2f} {top} (old first {f[top]['old_first']:.2f}), "
                                              f"control {f['control_random_read']['applied']:.2f}")

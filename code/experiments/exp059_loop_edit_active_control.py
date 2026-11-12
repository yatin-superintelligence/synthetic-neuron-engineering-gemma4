"""EXP-059: is the loop edit's effect specific? Replication at new seeds against a control of matched strength.

EXP-053 at Q4_0 lost 4 loops and gained 1, against no change for random directions. But a random unit direction carries about
1/1536 of a layer's write, so that control barely changes the model: 8 of 23 responses kept their exact length, against 1 of 23
for the edit. A weak control cannot show specificity. Here each control direction u is drawn inside the layer's own singular
directions so that ||u^T W|| matches ||v^T W|| for the loop direction v, and it is made orthogonal to v, so the same amount
of write is removed from an unrelated direction. New seeds 1 and 2 on the 15 loop contexts of the dev split, plus 8 controls at
Q4_0. Screens stop a response once it counts as a loop. Paired counts are reported by context; the test split stays unseen.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, loops as L  # noqa: E402
from sne.artifacts import LOCAL  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402
from sne.stats import mcnemar_exact  # noqa: E402

LAYERS = (14, 15, 19, 23)
LAM = -1.0
NAMES = ("remove_loop_dirs", "remove_matched_dirs")
SEEDS = (1, 2)


def build(exp_dir):
    """Child process, so the memory of the build returns to the system (see EXP-053)."""
    import torch
    from sne.artifacts import base_weights, build_adapters
    from sne.lora import direction_project
    vs = torch.load(os.path.join(ROOT, "artifacts_local", "EXP-040", "directions.pt"))
    torch.manual_seed(59)
    bw = base_weights("f16")
    matched, report = {}, {}
    for i in LAYERS:
        W = bw.get(i, "mlp.down_proj")
        v = vs[i].float()
        gain = float((v @ W).norm())
        U, S, _ = torch.linalg.svd(W, full_matrices=False)
        idx = torch.argsort((S - gain).abs())[:32]  # the singular directions whose gain is closest to the loop direction's
        c = torch.randn(len(idx)) * 1.0
        u = U[:, idx] @ (c / S[idx])  # weighting by 1/S keeps the gain close to the target when S varies across idx
        u = u - (u @ v) * v
        u = u / u.norm()
        u = u * 1.0
        matched[i] = u
        report[i] = dict(loop_gain=round(gain, 4), control_gain=round(float((u @ W).norm()), 4), overlap=round(float(u @ v), 6), rms_gain=round(float(S.pow(2).mean().sqrt()), 4))
    print("gains", report)
    json.dump(report, open(os.path.join(exp_dir, "gains.json"), "w", encoding="utf-8"), indent=1)

    def removal(dirs):
        def fn(e, b, prec):
            for i in LAYERS:
                direction_project(e, b.get(i, "mlp.down_proj"), i, "mlp.down_proj", dirs[i].float(), LAM)
            e.meta.update(layers=LAYERS, lam=LAM)
        return fn
    ex = SimpleNamespace(id="EXP-059", dir=exp_dir)
    for name, dirs in zip(NAMES, (vs, matched)):
        build_adapters(ex, name, removal(dirs))


if "--build" in sys.argv:
    build(sys.argv[sys.argv.index("--build") + 1])
    sys.exit(0)

with Experiment("EXP-059", "loop-edit-active-control", category="improvement", depends_on=["EXP-053"],
                motivation="EXP-053's control was far weaker than the edit, so its null result does not show that the edit is specific.",
                hypothesis="The loop edit lowers the loop rate at new seeds, and a matched-strength control does not.",
                config=dict(layers=LAYERS, lam=LAM, seeds=SEEDS, early_exit=True, split="dev", control="singular directions of matched gain, orthogonal to the loop direction, seed 59")) as ex:
    tok = tokenizer()
    subprocess.run([sys.executable, os.path.abspath(__file__), "--build", ex.dir], check=True)
    adapters = {n: {p: os.path.join(LOCAL, ex.id, f"{n}_{p}.gguf") for p in ("f16", "q4_0")} for n in NAMES}
    names = list(NAMES)
    cases = L.load("dev")[:2 if SMOKE else None]
    loops_only = [c for c in cases if c["kind"] == "loop"]
    subset = {"q4_0": loops_only + [c for c in cases if c["kind"] == "control"][:8], "f16": loops_only}
    out = {}
    for prec in ["q4_0", "f16"]:
        out[prec] = {}
        with LlamaServer(GGUF[prec], loras=[adapters[n][prec] for n in names], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            rows = {}
            for name in ["base"] + names:
                scales = {names.index(name): 1.0} if name != "base" else {}
                summ, r, texts = L.run(srv, tok, subset[prec], lora_scales=scales, seeds=SEEDS, early_exit=True)
                os.makedirs(os.path.join(LOCAL, ex.id), exist_ok=True)  # generated text can quote task content, so it stays out of git
                json.dump(texts, open(os.path.join(LOCAL, ex.id, f"texts_{prec}_{name}.json"), "w", encoding="utf-8"))
                rows[name] = {(x["case"], x["seed"]): x for x in r}
                out[prec][name] = summ
                ex.save_json(f"rows_{prec}_{name}.json", r)
                ex.log(prec, name, summ)
                ex.result(arms=out)
        keys = [k for k, x in rows["base"].items() if x["kind"] == "loop"]
        for name in names:
            lost = sum(rows["base"][k]["loop"] and not rows[name][k]["loop"] for k in keys)
            gained = sum(rows[name][k]["loop"] and not rows["base"][k]["loop"] for k in keys)
            out[prec][name]["paired"] = dict(pairs=len(keys), lost=lost, gained=gained, mcnemar_p=mcnemar_exact(lost, gained))
        ex.log(prec, "paired", {n: out[prec][n]["paired"] for n in names})
        ex.result(arms=out)
    q = out["q4_0"]
    ex.result(arms=out, headline="Q4_0 loops " + ", ".join(f"{k} {v['loop_rate_on_loop_contexts'][0]:.2f}" for k, v in q.items()) + f"; paired edit {q[names[0]]['paired']}, control {q[names[1]]['paired']}")

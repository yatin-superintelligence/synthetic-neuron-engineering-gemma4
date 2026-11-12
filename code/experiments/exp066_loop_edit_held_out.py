"""EXP-066: does the loop edit work on tasks it has never seen? The one held-out test of it (LoopBench test split).

EXP-053, EXP-059 and EXP-065 all used the dev split, where 3 tasks hold nearly all the loops, and the pattern was fitted on half of the
dev tasks. So those runs cannot show generality. This run uses the test split: 24 saved moments just before a real loop, from 14 tasks,
including FastAPI, which dev lacks. No task here was used to find the pattern, choose its strength, or choose its layers.
Fixed before the run: the full-removal edit (strength 1, layers 14, 15, 19, 23) as used in EXP-059 and in the Saturday Kaggle runs. Strength
1.5 looked slightly better on dev, but too few moments separate them, and strength 1 is the replicated version. The matched-strength
control is unchanged. Seeds 0 and 1 on loop moments, plus 12 normal moments at seed 0. Primary measure: loops across loop moments,
base against edit. Primary test: sign-flip across tasks, because runs within a task are not independent. Secondary: complete tool
calls, repetition score, tokens. Text is saved locally for reading. This split is used once and not again for this edit.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import SMOKE, loops as L  # noqa: E402
from sne.artifacts import LOCAL  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

NAMES = ("remove_loop_dirs", "remove_matched_dirs")
SEEDS = (0, 1)
ADAPTERS = {n: {p: os.path.join(LOCAL, "EXP-059", f"{n}_{p}.gguf") for p in ("f16", "q4_0")} for n in NAMES}


def task_test(base, arm, n_boot=20000, seed=66):
    """Change in loops per task, with a sign-flip test across tasks."""
    by = {}
    for k, b in base.items():
        t = k[0].split("/")[1]
        by[t] = by.get(t, 0) + int(b["loop"]) - int(arm[k]["loop"])
    nets = list(by.values())
    obs = sum(nets)
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in nets)) >= abs(obs) for _ in range(n_boot))
    return dict(tasks=len(nets), tasks_changed=sum(x != 0 for x in nets), net_loops_removed=obs, sign_flip_p=round(hits / n_boot, 4), by_task={k: v for k, v in by.items() if v})


with Experiment("EXP-066", "loop-edit-held-out", category="improvement", depends_on=["EXP-059", "EXP-065"],
                motivation="All evidence so far comes from 3 tasks of the dev split. A held-out test on 14 unseen tasks decides whether the effect is general.",
                hypothesis="The edit removes loops across several unseen tasks, and the control does not. If loops are removed in only 1 or 2 tasks, the effect is task-specific.",
                config=dict(split="test", seeds=SEEDS, strength=1.0, layers=[14, 15, 19, 23], primary="task-level sign-flip", early_exit=True)) as ex:
    tok = tokenizer()
    cases = L.load("test")[:2 if SMOKE else None]
    loops_only = [c for c in cases if c["kind"] == "loop"]
    controls = [c for c in cases if c["kind"] == "control"][:12]
    ex.log("loop moments", len(loops_only), "tasks", len({c["instance_id"] for c in loops_only}), "controls", len(controls))
    if not loops_only:
        raise RuntimeError("no states to test")
    os.makedirs(os.path.join(LOCAL, ex.id), exist_ok=True)
    out = {}
    for prec, subset in (("q4_0", loops_only + controls), ("f16", loops_only)):
        out[prec] = {}
        rows = {}
        with LlamaServer(GGUF[prec], loras=[ADAPTERS[n][prec] for n in NAMES], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            for name in ("base",) + NAMES:
                scales = {NAMES.index(name): 1.0} if name != "base" else {}
                summ, r, texts = L.run(srv, tok, subset, lora_scales=scales, seeds=SEEDS, early_exit=True)
                rows[name] = {(x["case"], x["seed"]): x for x in r if x["kind"] == "loop"}
                out[prec][name] = summ
                ex.save_json(f"rows_{prec}_{name}.json", r)
                json.dump(texts, open(os.path.join(LOCAL, ex.id, f"texts_{prec}_{name}.json"), "w", encoding="utf-8"))
                ex.log(prec, name, summ)
                ex.result(arms=out)
        for n in NAMES:
            out[prec][n]["task_level"] = task_test(rows["base"], rows[n])
            ex.log(prec, n, out[prec][n]["task_level"])
        ex.result(arms=out)
    q, f = out["q4_0"], out["f16"]
    ex.result(arms=out, headline=f"held out: Q4_0 loops base {q['base']['loop_rate_on_loop_contexts'][0]:.2f}, edit {q[NAMES[0]]['loop_rate_on_loop_contexts'][0]:.2f}, control {q[NAMES[1]]['loop_rate_on_loop_contexts'][0]:.2f}, task p {q[NAMES[0]]['task_level']['sign_flip_p']}; "
                                 f"F16 {f['base']['loop_rate_on_loop_contexts'][0]:.2f}, {f[NAMES[0]]['loop_rate_on_loop_contexts'][0]:.2f}, {f[NAMES[1]]['loop_rate_on_loop_contexts'][0]:.2f}, task p {f[NAMES[0]]['task_level']['sign_flip_p']}")

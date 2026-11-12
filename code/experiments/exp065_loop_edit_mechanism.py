"""EXP-065: is the loop edit causal, and which part does the work? Dose, sign reversal, single layers, prior art (LoopBench dev).

The gap audit found that lambda = -1 and these four layers were never justified, and that causality rests on one direction of
change. Arms, all as exact LoRA adapters:
  dose:        lambda -0.5, -1.5, -2 at all four layers. Causation predicts a graded response.
  reversal:    lambda +0.5 and +1 at all four layers, which adds the direction. If it carries looping, loops should rise, including on
               controls. Removal alone cannot show this.
  single:      lambda -1 at each of layers 14, 15, 19 and 23 alone, to see which layer does the work.
  prior art:   the two-neuron E2B edit of arXiv 2606.13705 (L10:3513 scaled by -0.8, L12:3838 scaled by 3.0), found on list prompts,
               tested here for the first time on real agent states.
Reference values come from a base arm run here. Loop contexts plus 8 controls at seed 0 on Q4_0. F16 repeats base, the best
removal, reversal +1 and prior art on loop contexts. Reported by context and by task, because EXP-053 and EXP-059 showed that
nearly all loops sit in 3 tasks of this split. Only dev is used.
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

ALL = (14, 15, 19, 23)
DIR_ARMS = {"lam_m0.5": (ALL, -0.5), "lam_m1.5": (ALL, -1.5), "lam_m2": (ALL, -2.0), "lam_p0.5": (ALL, 0.5), "lam_p1": (ALL, 1.0),
            "only_14": ((14,), -1.0), "only_15": ((15,), -1.0), "only_19": ((19,), -1.0), "only_23": ((23,), -1.0)}
NAMES = list(DIR_ARMS) + ["prior_art_2neuron"]
F16_ARMS = ["lam_m1.5", "lam_p1", "prior_art_2neuron"]


def build(exp_dir):
    """Child process, so the build's memory returns to the system (see EXP-053)."""
    import torch
    from sne.artifacts import build_adapters
    from sne.lora import direction_project, neuron_scale
    vs = torch.load(os.path.join(ROOT, "artifacts_local", "EXP-040", "directions.pt"))
    ex = SimpleNamespace(id="EXP-065", dir=exp_dir)

    def direction(layers, lam):
        def fn(e, bw, prec):
            for i in layers:
                direction_project(e, bw.get(i, "mlp.down_proj"), i, "mlp.down_proj", vs[i].float(), lam)
            e.meta.update(layers=layers, lam=lam)
        return fn

    def prior(e, bw, prec):
        neuron_scale(e, bw.get(10, "mlp.down_proj"), 10, [3513], -0.8)
        neuron_scale(e, bw.get(12, "mlp.down_proj"), 12, [3838], 3.0)
        e.meta.update(source="arXiv 2606.13705")
    names = NAMES[:2] + NAMES[-1:] if SMOKE else NAMES
    for n in names:
        build_adapters(ex, n, prior if n == "prior_art_2neuron" else direction(*DIR_ARMS[n]))


if "--build" in sys.argv:
    build(sys.argv[sys.argv.index("--build") + 1])
    sys.exit(0)

with Experiment("EXP-065", "loop-edit-mechanism", category="improvement", depends_on=["EXP-059"],
                motivation="A reviewer will ask why this dose and these layers, whether the effect reverses, and how it compares with the only "
                           "published Gemma loop edit. None of these had an answer.",
                hypothesis="Loop rate falls with dose; adding the direction raises loops, including on controls; one or two layers carry most "
                           "of the effect; the prior-art edit, built on list prompts, does less on agent states.",
                config=dict(arms={k: list(v) for k, v in DIR_ARMS.items()}, prior_art="L10:3513 x -0.8, L12:3838 x 3.0", seeds=[0], early_exit=True, split="dev")) as ex:
    tok = tokenizer()
    subprocess.run([sys.executable, os.path.abspath(__file__), "--build", ex.dir], check=True)
    names = NAMES[:2] + NAMES[-1:] if SMOKE else NAMES
    adapters = {n: {p: os.path.join(LOCAL, ex.id, f"{n}_{p}.gguf") for p in ("f16", "q4_0")} for n in names}
    cases = L.load("dev")[:2 if SMOKE else None]
    loops_only = [c for c in cases if c["kind"] == "loop"]
    out = {}
    plan = {"q4_0": (loops_only + [c for c in cases if c["kind"] == "control"][:8], names), "f16": (loops_only, [n for n in F16_ARMS if n in names] or names[:1])}
    for prec, (subset, arms) in plan.items():
        out[prec] = {}
        with LlamaServer(GGUF[prec], loras=[adapters[n][prec] for n in names], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            for name in ["base"] + arms:
                scales = {names.index(name): 1.0} if name != "base" else {}
                summ, rows, texts = L.run(srv, tok, subset, lora_scales=scales, seeds=(0,), early_exit=True)
                summ["task_net_loops"] = {}
                for r in rows:
                    t = r["case"].split("/")[1]
                    summ["task_net_loops"][t] = summ["task_net_loops"].get(t, 0) + int(r["loop"])
                out[prec][name] = summ
                ex.save_json(f"rows_{prec}_{name}.json", rows)
                os.makedirs(os.path.join(LOCAL, ex.id), exist_ok=True)
                json.dump(texts, open(os.path.join(LOCAL, ex.id, f"texts_{prec}_{name}.json"), "w", encoding="utf-8"))
                ex.log(prec, name, {k: summ[k] for k in ("loop_rate_on_loop_contexts", "loop_rate_on_controls", "tool_call_rate_on_controls", "mean_tokens")})
                ex.result(arms=out)
    q = out["q4_0"]
    ex.result(arms=out, headline="Q4_0 loops " + ", ".join(f"{k} {v['loop_rate_on_loop_contexts'][0]:.2f}" for k, v in q.items()))

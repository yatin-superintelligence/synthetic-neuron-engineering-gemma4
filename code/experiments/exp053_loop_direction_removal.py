"""EXP-053: does removing the loop direction stop loops? Causal test of EXP-040 (LoopBench dev split).

EXP-040 found a direction in MLP writes that separates loop tokens, strongest at layers 14, 15, 19 and 23. Each is removed by
a rank-1 edit, W' = (I - v v^T) W on down_proj, compiled exactly to LoRA for each precision. Control: the same edit along
random unit directions in the same layers, so any change caused by perturbing those layers shows up there too. Replay: loop
contexts plus the first 8 controls, one seed, official sampling and limits; a response stops once it counts as a loop. The
directions were fitted on half of the dev tasks, so dev evaluation is mildly optimistic; the test split stays unseen.
Loop contexts are slow here, so F16 runs on loop contexts only. Loop rate is read together with tool-call rate on controls,
because an edit that stops loops by breaking generation is not a fix.
"""
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

LAYERS = (14, 15, 19, 23)
LAM = -1.0
NAMES = ("remove_loop_dirs", "remove_random_dirs")


def build(exp_dir):
    """Runs in a child process: the adapter build peaked at 21 GB of private memory and kept 13 GB, which left the machine
    under 1 GB of commit space during the replays. When the child exits, all of that returns to the system."""
    import torch
    from sne.artifacts import build_adapters
    from sne.lora import direction_project
    vs = torch.load(os.path.join(ROOT, "artifacts_local", "EXP-040", "directions.pt"))
    torch.manual_seed(53)
    rnd = {i: torch.nn.functional.normalize(torch.randn_like(vs[i].float()), dim=0) for i in LAYERS}
    print("random direction overlap with the loop direction", {i: round(float(rnd[i] @ vs[i].float()), 4) for i in LAYERS})

    def removal(dirs):
        def fn(e, bw, prec):
            for i in LAYERS:
                direction_project(e, bw.get(i, "mlp.down_proj"), i, "mlp.down_proj", dirs[i].float(), LAM)
            e.meta.update(layers=LAYERS, lam=LAM)
        return fn
    ex = SimpleNamespace(id="EXP-053", dir=exp_dir)
    for name, dirs in zip(NAMES, (vs, rnd)):
        build_adapters(ex, name, removal(dirs))


if "--build" in sys.argv:
    build(sys.argv[sys.argv.index("--build") + 1])
    sys.exit(0)

with Experiment("EXP-053", "loop-direction-removal", category="improvement", depends_on=["EXP-040"],
                motivation="EXP-040 is descriptive. Only an intervention can show that the direction carries looping, as "
                           "Takehi et al. showed for Qwen. Loops took 31% of tokens in EXP-037 and preceded 10 of 15 context errors.",
                hypothesis="Removing the direction lowers the loop rate on loop contexts more than removing random directions, "
                           "without lowering tool-call rate on controls.",
                config=dict(layers=LAYERS, lam=LAM, seeds=[0], early_exit=True, split="dev", control="random unit directions in the same layers, seed 53")) as ex:
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
            for name in ["base"] + names:
                scales = {names.index(name): 1.0} if name != "base" else {}
                summ, rows, _ = L.run(srv, tok, subset[prec], lora_scales=scales, seeds=(0,), early_exit=True)
                out[prec][name] = summ
                ex.save_json(f"rows_{prec}_{name}.json", rows)
                ex.log(prec, name, summ)
                ex.result(arms=out)
    q = out["q4_0"]
    ex.result(arms=out, headline="Q4_0 loop rate " + ", ".join(f"{k} {v['loop_rate_on_loop_contexts'][0]:.2f}" for k, v in q.items()) +
                                 "; control tool calls " + ", ".join(f"{k} {v['tool_call_rate_on_controls'][0]:.2f}" for k, v in q.items()))

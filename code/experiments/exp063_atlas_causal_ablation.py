"""EXP-063: are the atlas groups responsible for what they respond to? Switch each one off and see what breaks.

EXP-062 found neuron groups that respond to writing old_string, tool-call syntax and whitespace. Response is not responsibility. Each
group is ablated (down_proj column set to zero, an exact LoRA) and compared with random neurons drawn in the same counts from the same
layers. Predictions, written before the run: old_string neurons off should lower the apply rate on real edit states more than
random; syntax neurons off should lower complete tool calls; whitespace neurons off should raise broken edits. A group that does not
do this, or a random set that does the same, means the group is a correlate. Dev edit states at Q4_0, seed 0; F16 repeats base, the
old_string group and its random control. Reported by task. Ablation is a test, not a proposed edit, so the competition's rank limit
does not apply here.
"""
import glob
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, editbench as E, loops as L  # noqa: E402
from sne.artifacts import LOCAL  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

D62 = glob.glob(os.path.join(ROOT, "experiments", "EXP-062_*"))[0]
ARMS = {"old_string_on": (13, 9), "old_string_off": (5, 6), "syntax_on": (3, 12, 1, 4), "syntax_off": (0, 14, 7), "whitespace_on": (11,), "system_on": (2,)}
MAX_PROMPT, N_STATES = 12000, 32
F16_ARMS = ["old_string_on", "rand_old_string_on"]


def neuron_sets():
    groups = {g["group"]: g for g in json.load(open(os.path.join(D62, "groups.json"), encoding="utf-8"))}
    widths = [6144] * 15 + [12288] * 20
    sets = {}
    for name, ids in ARMS.items():
        sel = [tuple(n) for g in ids for n in groups[g]["neurons"]]
        sets[name] = sel
        if name in ("old_string_on", "syntax_on"):  # matched random neurons: same count per layer, none from the group
            rng = random.Random(63)
            taken = set(sel)
            counts = {}
            for l, _ in sel:
                counts[l] = counts.get(l, 0) + 1
            sets["rand_" + name] = [(l, j) for l, c in counts.items() for j in rng.sample([x for x in range(widths[l]) if (l, x) not in taken], c)]
    return sets


def build(exp_dir):
    """Child process, so the build's memory returns to the system (see EXP-053)."""
    from sne.artifacts import build_adapters
    from sne.lora import neuron_scale
    ex = SimpleNamespace(id="EXP-063", dir=exp_dir)
    sets = neuron_sets()

    def ablate(sel):
        def fn(e, bw, prec):
            by = {}
            for l, j in sel:
                by.setdefault(l, []).append(j)
            for l, idx in sorted(by.items()):
                neuron_scale(e, bw.get(l, "mlp.down_proj"), l, idx, 0.0)
            e.meta.update(n_neurons=len(sel))
        return fn
    names = list(sets)[:2] if SMOKE else list(sets)
    for n in names:
        build_adapters(ex, n, ablate(sets[n]))


if "--build" in sys.argv:
    build(sys.argv[sys.argv.index("--build") + 1])
    sys.exit(0)

with Experiment("EXP-063", "atlas-causal-ablation", category="localization", depends_on=["EXP-062"],
                motivation="The atlas shows which neurons respond to which activity. Only intervention shows which are responsible for it.",
                hypothesis="Each group's ablation damages its own function more than matched random neurons do.",
                config=dict(arms={k: list(v) for k, v in ARMS.items()}, states=N_STATES, max_prompt=MAX_PROMPT, split="dev", seed=0)) as ex:
    tok = tokenizer()
    subprocess.run([sys.executable, os.path.abspath(__file__), "--build", ex.dir], check=True)
    sets = neuron_sets()
    names = list(sets)[:2] if SMOKE else list(sets)
    ex.save_json("sets.json", {k: len(v) for k, v in sets.items()})
    adapters = {n: {p: os.path.join(LOCAL, ex.id, f"{n}_{p}.gguf") for p in ("f16", "q4_0")} for n in names}
    cases = E.short_cases(tok, E.load("dev", per_run_task=2), MAX_PROMPT)[:2 if SMOKE else N_STATES]
    ex.log("states", len(cases), "tasks", len({c["instance_id"] for c in cases}))
    if not cases:
        raise RuntimeError("no states to test")
    out = {}
    # resume: an arm whose rows and summary survive from an interrupted run is not run again (same states, seed and adapters)
    rp = os.path.join(ex.dir, "results_resume.json")
    prev = json.load(open(rp, encoding="utf-8")).get("results", {}).get("arms", {}) if os.path.exists(rp) and not SMOKE else {}
    for prec, arms in (("q4_0", names), ("f16", [n for n in F16_ARMS if n in names] or names[:1])):
        out[prec] = {}
        with LlamaServer(GGUF[prec], loras=[adapters[n][prec] for n in names], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            for name in ["base"] + arms:
                if name in prev.get(prec, {}) and os.path.exists(os.path.join(ex.dir, f"rows_{prec}_{name}.json")):
                    out[prec][name] = prev[prec][name]
                    ex.log(prec, name, "kept from the interrupted run")
                    continue
                scales = {names.index(name): 1.0} if name != "base" else {}
                summ, rows = E.run(srv, tok, cases, lora_scales=scales, seeds=(0,))
                out[prec][name] = summ
                ex.save_json(f"rows_{prec}_{name}.json", rows)
                ex.log(prec, name, summ["all"], summ["mean_tokens"])
                ex.result(arms=out)
    q = out["q4_0"]
    ex.result(arms=out, headline="Q4_0 apply rate " + ", ".join(f"{k} {v['all']['apply_rate']:.2f}" for k, v in q.items()))

"""EXP-097b: EXP-097's escape-blind values at F16 on real edit states (EditBench), the bf16 counterpart of EXP-097's W4A16 test.

EXP-097 (Colab, official W4A16 on vLLM, injected-bug bench) cut answers with a copied backslash-quote from 32 to 22 of 257 held-out
states (control 27), while applying answers fell (99 to 88; control 91). This run asks whether the same edit acts the same way at F16
(llama.cpp) and on the states of real official runs: EditBench replays the agent state before each recorded edit_file call of the
official traces, scored against the file as it stood at that step (sne.editbench).

Pre-registered (D-080, before any result): the selected EXP-097 edit (layers 13-14, alpha 0.5) and its control, read from EXP-097's
adapters, built as F16 GGUF LoRAs; EditBench states of prompts up to 12,000 tokens, one per task and run (seeded), seed 0, the
official sampling. Arms: unedited, edit, control, one llama-server. Outcomes per arm: replayed edits whose old_string carries a
backslash, the apply rate of edit calls, not-found failures. Paired by state: exact McNemar on backslash-carrying edits and on
applied edits, edit against unedited and against control. Expected from EXP-097: fewer backslash edits than both, apply rate not
higher.
"""
import os
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, editbench as E, loops as L  # noqa: E402
from sne.artifacts import LOCAL  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402
from sne.stats import mcnemar_exact  # noqa: E402

SRC = os.path.join(ROOT, "artifacts_local", "EXP-097")
NAMES = {"edit": "L13-14_a0.5", "control": "control"}
MAX_PROMPT = 12000


def build(exp_dir):
    """Child process: GGUF LoRAs from EXP-097's PEFT adapters (v_proj, rank 1 per layer)."""
    from safetensors.torch import load_file
    from sne.artifacts import build_adapters
    ex = SimpleNamespace(id="EXP-097b", dir=exp_dir)

    def from_peft(folder):
        sd = load_file(os.path.join(SRC, folder, "adapter_model.safetensors"))

        def fn(e, bw, prec):
            layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in sd})
            for i in layers:
                base = f"base_model.model.model.language_model.layers.{i}.self_attn.v_proj"
                e.add(i, "self_attn.v_proj", sd[f"{base}.lora_B.weight"].float(), sd[f"{base}.lora_A.weight"].float())
            e.meta.update(layers=layers, source=folder)
        return fn
    for n, folder in NAMES.items():
        build_adapters(ex, n, from_peft(folder), precisions=("f16",))


if "--build" in sys.argv:
    build(sys.argv[sys.argv.index("--build") + 1])
    sys.exit(0)

with Experiment("EXP-097b", "escape-blind-editbench-f16", category="improvement", depends_on=["EXP-097"],
                motivation="bf16 counterpart of EXP-097 and a transfer test: the escape-blind value edit at F16 on real edit states from the "
                           "official traces.",
                hypothesis="At F16 on EditBench, the escape-blind edit lowers replayed edits whose old_string carries a backslash against the "
                           "unedited model and its control (exact McNemar by state), without raising the apply rate.",
                config=dict(edit=NAMES["edit"], control=NAMES["control"], max_prompt=MAX_PROMPT, per_run_task=1, seed=0, precision="f16")) as ex:
    tok = tokenizer()
    subprocess.run([sys.executable, os.path.abspath(__file__), "--build", ex.dir], check=True)
    adapters = [os.path.join(LOCAL, ex.id, f"{n}_f16.gguf") for n in NAMES]
    cases = E.short_cases(tok, E.load(None, per_run_task=1), MAX_PROMPT)[:4 if SMOKE else None]
    ex.log("states", len(cases), "tasks", len({c["instance_id"] for c in cases}), "originally applied", sum(c["ok"] for c in cases))
    if not cases:
        raise RuntimeError("no states to test")
    out, rows = {}, {}
    with LlamaServer(GGUF["f16"], loras=adapters, parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                     log_path=os.path.join(ex.dir, "server_f16.log")) as srv:
        for k, name in enumerate(["base"] + list(NAMES)):
            scales = {list(NAMES).index(name): 1.0} if name != "base" else {}
            summ, rws = E.run(srv, tok, cases, lora_scales=scales, seeds=(0,))
            out[name], rows[name] = summ, {r["case"]: r for r in rws}
            ex.save_json(f"rows_{name}.json", rws)
            ex.log(name, summ["all"], summ["mean_tokens"])
            ex.result(arms=out)
    common = sorted(set.intersection(*(set(v) for v in rows.values())))

    def flag(name, key):
        return [bool(rows[name][c]["tool"] == "edit_file" and (rows[name][c][key] if key == "backslash" else rows[name][c]["outcome"] == "applied"))
                for c in common]
    tests = {}
    for other in ("base", "control"):
        for key in ("backslash", "applied"):
            a, b = flag("edit", key), flag(other, key)
            x = sum(1 for i, j in zip(a, b) if i and not j)
            y = sum(1 for i, j in zip(a, b) if j and not i)
            tests[f"edit_vs_{other}_{key}"] = dict(edit_only=x, other_only=y, p=round(mcnemar_exact(x, y), 4))
    tot = {n: dict(backslash=sum(flag(n, "backslash")), applied=sum(flag(n, "applied")), edit_calls=out[n]["all"]["edit_calls"]) for n in rows}
    ex.result(arms=out, states=len(common), totals=tot, tests=tests,
              headline=f"F16 EditBench ({len(common)} states): backslash edits unedited {tot['base']['backslash']}, edit {tot['edit']['backslash']}, "
                       f"control {tot['control']['backslash']}; applied {tot['base']['applied']}, {tot['edit']['applied']}, {tot['control']['applied']}")

"""EXP-041: why half of edit_file calls fail, and whether the cause can be shown by intervention (EditBench dev split).

Census: every edit_file call in official Kaggle traces is walked against the file as it stood at that step and classified
with the official matcher. Argument order is then tested directly on the dev split. The tool schema is sorted, so the model
writes new_string before old_string. From the recorded thought and file path, one branch must write old_string first and
the other new_string first. The prefix is identical up to that token, so order is the only difference. Two seeds; paired
test on seed 0, bootstrap over tasks. Local prompt processing runs at about 130 tokens/s on long contexts, so the test uses
states with full prompts of at most 9000 tokens, which are the earlier edits of each task. Prompt rules are tested on real
tasks on Kaggle instead (D-029). Committed outputs hold ids and counts only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, editbench as E, loops as L  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

DIRS = {"exp036": os.path.join(ROOT, "artifacts_local", "kaggle", "exp029", "results", "traces")}
d37 = os.path.join(ROOT, "artifacts_local", "kaggle", "exp037", "results", "traces")
if os.path.isdir(d37):
    DIRS["exp037"] = d37
ORDER_SEEDS = (0, 1)
MAX_PROMPT = 9000

with Experiment("EXP-041", "edit-failure-replay", category="localization", depends_on=["EXP-036"],
                motivation="50% of edit_file calls failed in EXP-036, 11 of 58 tasks never applied an edit, and none of those "
                           "resolved. Reading the traces suggests two structural causes that a weight edit could address.",
                hypothesis="Forcing old_string first raises the apply rate over the new_string-first branch.",
                config=dict(traces=list(DIRS), split="dev", order_seeds=ORDER_SEEDS, max_prompt_tokens=MAX_PROMPT, slots=L.SLOTS, per_run_task=4)) as ex:
    summary = {}
    for run, d in DIRS.items():
        rows = E.census(d)
        ex.save_json(f"census_{run}.json", rows)
        summary[run] = E.summarise(rows)
        ex.log(run, summary[run])
    built = E.build(DIRS)
    ex.log("cases", len(built), "dev", sum(c["split"] == "dev" for c in built))
    tok = tokenizer()
    cases = E.short_cases(tok, E.load("dev", per_run_task=4), MAX_PROMPT)
    ex.log("states tested", len(cases), "tasks", len({c["instance_id"] for c in cases}), "originally failed", sum(not c["ok"] for c in cases))
    if not cases:
        raise RuntimeError("no states to test")
    out = {}
    for prec in ["q4_0", "f16"]:
        out[prec] = {}
        with LlamaServer(GGUF[prec], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            rows, order = E.order_branches(srv, tok, cases, seeds=ORDER_SEEDS)
            out[prec]["argument_order"] = order
            ex.save_json(f"rows_{prec}_order.json", rows)
            ex.log(prec, "argument_order", order)
            ex.result(census=summary, replay=out)
    o = out["q4_0"]["argument_order"]
    f = out["f16"]["argument_order"]
    ex.result(census=summary, replay=out,
              headline=f"forced order: Q4_0 {o['rate_old_first']:.2f} old first vs {o['rate_new_first']:.2f} new first on {o['pairs']} pairs, gain CI {o['task_bootstrap_ci']}; "
                       f"F16 {f['rate_old_first']:.2f} vs {f['rate_new_first']:.2f}, gain CI {f['task_bootstrap_ci']}")

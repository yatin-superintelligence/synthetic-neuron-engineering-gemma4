"""EXP-052: forced argument order on states whose recorded edit failed (EditBench dev split).

EXP-041 tested 31 short states, of which only 5 had failed, so it says little about where a fix is needed: on Q4_0 the
applied states applied 49 of 52 times in either order, and the failed states 2 of 10 old first against 0 of 10 new first.
This run tests the population the hypothesis is about: failures that order could plausibly cause. Kinds kept: text mostly
real with a changed line, identical old and new, invented text, real lines in a broken block, over-escaped. Stale and
ambiguous failures are left out, since writing old_string first cannot fix either. Full prompts of at most 16000 tokens,
which covers most failed states, at about 100 s per prompt. Design otherwise as EXP-041: identical prefix up to the first
argument key, two seeds, paired test on seed 0, bootstrap over tasks. Only the dev split is used.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import editbench as E, loops as L  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

SEEDS = (0, 1)
MAX_PROMPT = 16000
KINDS = ("mostly_real_altered", "identical", "invented", "lines_real_block_not", "over_escaped")

with Experiment("EXP-052", "order-on-failed-states", category="localization", depends_on=["EXP-041"],
                motivation="EXP-041 was short of failed states, and the observed 9 of 10 against 91 of 207 may be selection. This "
                           "decides whether the order neuron (EXP-042) is worth building.",
                hypothesis="Writing old_string first turns failed edits into applied ones more often than the new-first branch. "
                           "If the gain CI includes zero, H-4's order cause is dropped.",
                config=dict(split="dev", seeds=SEEDS, max_prompt_tokens=MAX_PROMPT, kinds=KINDS, slots=L.SLOTS, per_run_task=4)) as ex:
    tok = tokenizer()
    by = {}
    for c in E.load("dev", per_run_task=10 ** 6):
        if c["kind"] in KINDS:
            by.setdefault((c["run"], c["instance_id"]), []).append(c)
    cases = [c for k in sorted(by) for c in by[k][:4]]
    cases = E.short_cases(tok, cases, MAX_PROMPT)
    ex.log("states", len(cases), "tasks", len({c["instance_id"] for c in cases}), "kinds", {k: sum(c["kind"] == k for c in cases) for k in KINDS})
    if not cases:
        raise RuntimeError("no states to test")
    out = {}
    for prec in ["q4_0", "f16"]:
        with LlamaServer(GGUF[prec], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            rows, order = E.order_branches(srv, tok, cases, seeds=SEEDS)
        out[prec] = order
        ex.save_json(f"rows_{prec}.json", rows)
        ex.log(prec, order)
        ex.result(order=out)
    q, f = out["q4_0"], out["f16"]
    ex.result(order=out, headline=f"failed states, forced order: Q4_0 {q['rate_old_first']:.2f} old first vs {q['rate_new_first']:.2f} new first on {q['pairs']} pairs, gain CI {q['task_bootstrap_ci']}; "
                                 f"F16 {f['rate_old_first']:.2f} vs {f['rate_new_first']:.2f}, gain CI {f['task_bootstrap_ci']}")

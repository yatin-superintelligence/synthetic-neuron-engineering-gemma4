"""EXP-038: census of thinking loops in official runs, LoopBench v1 build, and baseline replay at F16 and Q4_0.

Traces: official Kaggle E2B runs (EXP-036, plus EXP-037 when available), stored locally because they contain Competition
Data. Replay of the dev split: each pre-loop context at three seeds and each control once, thinking on, official sampling
(T=0.2, top_p 0.95), official limits: 32768 context, thinking forced closed at 4096 tokens as vLLM does, 16384 output.
The test split is held back for the final comparison. Committed outputs contain task ids and scores only.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, loops as L  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

DIRS = {"exp036": os.path.join(ROOT, "artifacts_local", "kaggle", "exp029", "results", "traces")}
d37 = os.path.join(ROOT, "artifacts_local", "kaggle", "exp037", "results", "traces")
if os.path.isdir(d37):
    DIRS["exp037"] = d37
SEEDS = (0, 1, 2)
LOCAL = os.path.join(ROOT, "artifacts_local", "EXP-038")

with Experiment("EXP-038", "loop-census-and-replay", category="localization", depends_on=["EXP-036"],
                motivation="H-3: E2B falls into sentence-level thinking loops on real tasks. A census fixes the rate; "
                           "LoopBench replays the pre-loop states so any fix can be screened in minutes on real contexts.",
                hypothesis="Pre-loop contexts reproduce loops far more often than control contexts, and loops are less "
                           "frequent but still present at F16.",
                config=dict(traces=list(DIRS), seeds=SEEDS, split="dev", temperature=0.2, top_p=0.95, thinking_budget=L.THINK_BUDGET,
                            max_output=L.MAX_OUTPUT, slot_ctx=L.SLOT_CTX, slots=L.SLOTS,
                            thresholds=dict(min_repeats=L.LOOP_MIN_REPEATS, min_rep3=L.LOOP_MIN_REP3))) as ex:
    summary = {}
    for run, d in DIRS.items():
        rows = L.census(d)
        ex.save_json(f"census_{run}.json", rows)
        lt = [r for r in rows if r["n_loops"]]
        ls = [x for r in rows for x in r["loops"]]
        summary[run] = dict(tasks=len(rows), tasks_with_loop=len(lt), loops=len(ls), loops_after_tool_error=sum(x["prev_tool_error"] for x in ls),
                            median_onset_char=sorted(x["onset_char"] or 0 for x in ls)[len(ls) // 2] if ls else None)
        ex.log(run, summary[run])
    cases = L.build(DIRS)
    ex.log("cases", {k: sum(c["kind"] == k for c in cases) for k in ("loop", "control")}, "tasks", len({c["instance_id"] for c in cases}))
    if not any(c["kind"] == "loop" for c in cases):
        raise RuntimeError("no loop contexts found")
    tok = tokenizer()
    cases = L.load("dev")
    out = {}
    for prec in ["q4_0", "f16"]:
        with LlamaServer(GGUF[prec], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            summ, rows, texts = L.run(srv, tok, cases, seeds=SEEDS)
        out[prec] = summ
        ex.save_json(f"replay_{prec}.json", rows)
        os.makedirs(LOCAL, exist_ok=True)  # generated text can quote task content, so it stays out of git
        json.dump(texts, open(os.path.join(LOCAL, f"replay_{prec}.json"), "w", encoding="utf-8"))
        ex.log(prec, summ)
        ex.result(census=summary, replay=out)
    ex.result(census=summary, replay=out,
              headline=" | ".join(f"{p}: loop contexts {v['loop_rate_on_loop_contexts'][0]:.2f}, controls {v['loop_rate_on_controls'][0]:.2f}" for p, v in out.items()))

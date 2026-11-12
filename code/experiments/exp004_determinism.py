"""EXP-004: is greedy decoding on the engine deterministic, and does slot parallelism change outputs?

Paired McNemar tests assume an arm's outcome per problem is a fixed function of the model. With continuous
batching, floating-point reduction order can depend on which requests share a batch, so outputs may differ between
runs. We measure: (1) run-to-run identity at parallel=8 on MBPP+ dev; (2) parallel=8 vs parallel=1 on the first 60
dev problems. Both precisions. This sets the minimum effect size we can attribute to an edit.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne.engine import LlamaServer  # noqa: E402
from sne.evaluate import run_suite  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, load_suite, tokenizer  # noqa: E402
from sne.stats import paired_compare  # noqa: E402

with Experiment("EXP-004", "determinism-and-batching", category="infrastructure", depends_on=["EXP-001"],
                motivation="Quantifies the evaluation noise floor that every edit comparison must exceed.",
                hypothesis="Run-to-run outputs at fixed parallelism are mostly identical; changing parallelism alters "
                           "a few percent of generations but pass-rate changes stay within +-1-2 points.",
                config=dict(suite="mbpp_dev", subset_parallel1=60)) as ex:
    tok = tokenizer()
    tasks = load_suite("mbpp_dev")
    out = {}
    for prec in ["f16", "q4_0"]:
        runs = {}
        with LlamaServer(GGUF[prec], parallel=8, log_path=os.path.join(ex.dir, f"server_{prec}_p8.log")) as srv:
            for r in ("run1", "run2"):
                s, rows = run_suite(srv, tok, tasks)
                runs[r] = rows
        with LlamaServer(GGUF[prec], parallel=1, port=8092,
                         log_path=os.path.join(ex.dir, f"server_{prec}_p1.log")) as srv:
            s, rows = run_suite(srv, tok, tasks[:60])
            runs["p1"] = rows
        a, b, c = runs["run1"], runs["run2"], runs["p1"]
        ident = sum(x["response"] == y["response"] for x, y in zip(a, b))
        ident_p1 = sum(x["response"] == y["response"] for x, y in zip(a[:60], c))
        pc = paired_compare({x["task_id"]: x["passed"] for x in a}, {x["task_id"]: x["passed"] for x in b})
        pc1 = paired_compare({x["task_id"]: x["passed"] for x in a[:60]}, {x["task_id"]: x["passed"] for x in c})
        out[prec] = dict(identical_run1_run2=ident, n=len(a), identical_p8_vs_p1=ident_p1, n_p1=60,
                         pass_run1=pc["base"], pass_run2=pc["treat"], flips_run1_run2=pc["gained"] + pc["lost"],
                         pass_p8_subset=pc1["base"], pass_p1_subset=pc1["treat"], flips_p8_p1=pc1["gained"] + pc1["lost"])
        ex.log(prec, out[prec])
        ex.save_json(f"responses_{prec}.json", {k: [x["response"] for x in v] for k, v in runs.items()})
    ex.result(**out, headline="; ".join(f"{p}: run-run identical {v['identical_run1_run2']}/{v['n']}, p8-vs-p1 "
                                        f"identical {v['identical_p8_vs_p1']}/60, outcome flips {v['flips_run1_run2']}"
                                        for p, v in out.items()))

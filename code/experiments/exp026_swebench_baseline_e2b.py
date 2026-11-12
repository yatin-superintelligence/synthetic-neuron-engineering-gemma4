"""EXP-026: first real-repository baseline. E2B QAT at Q4_0 and F16 on the 60 SWE-bench Verified dev issues.

Agent: src/sne/sweagent.py (competition tools, native tool syntax, 100 turns, 24k token context, 8192 output tokens, greedy).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import sweagent as A  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

WORKERS = 4
with Experiment("EXP-026", "swebench-baseline-e2b", category="baseline", depends_on=["EXP-025", "EXP-003"],
                motivation="Decides whether E2B can do real SWE at all, and which stage of the agent funnel loses tasks. "
                           "Tests H-1 (edit_file failures from inexact copying) on real code.",
                hypothesis="E2B resolves under 15% of dev issues. Most tasks fail before a correct patch: it reaches the "
                           "gold file in most tasks but many edit_file calls fail on old_string.",
                config=dict(split="dev", workers=WORKERS, max_turns=100, ctx_budget=24000, max_tokens=8192,
                            ctx_per_slot=32768)) as ex:
    A.ensure_docker(ex.log)
    tok = tokenizer()
    tasks = A.load_subset("dev")
    A.pull(tasks, ex.log)
    out = {}
    for prec in ["q4_0", "f16"]:
        with LlamaServer(GGUF[prec], parallel=WORKERS, ctx_per_slot=32768,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            rows = A.run_many(srv, tok, tasks, workers=WORKERS, log=ex.log)
        with open(os.path.join(ex.dir, f"outputs_{prec}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out[prec] = A.summarize(rows)
        ex.log(prec, out[prec])
        ex.result(**out)
    ex.result(**out, headline=" | ".join(f"{p}: resolved {v['resolved']:.3f}, gold file {v['visited_gold']:.2f}, patch "
                                         f"{v['patch_nonempty']:.2f}, edit ok {v['edit_success_rate']}" for p, v in out.items()))
    A.stop_docker()

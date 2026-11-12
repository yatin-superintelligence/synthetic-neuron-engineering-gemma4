"""EXP-011: baseline of the execution-feedback repair loop (src/sne/agent.py) at F16 and Q4_0.

HumanEvalFix (visible = docstring examples) and MBPP+ dev (visible = the one assert shown in the prompt),
up to 3 turns, scored on hidden tests. Measures how much a small model gains from tool feedback, and how often
it is falsely confident (passes visible tests, fails hidden ones).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne.agent import repair_loop  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, load_suite, tokenizer  # noqa: E402

PAR = int(os.environ.get("SNE_PARALLEL", "8"))
with Experiment("EXP-011", "agent-repair-loop-baseline", category="baseline", depends_on=["EXP-010"],
                motivation="The competition evaluates multi-turn agents. A single-shot benchmark cannot show whether an "
                           "edit changes how the model uses tool feedback. This loop is the cheapest faithful proxy.",
                hypothesis="Feedback turns raise hidden-test pass rate by several points over turn 1; some solutions "
                           "pass visible tests but fail hidden tests (false confidence).",
                config=dict(max_turns=3, suites=["humanevalfix", "mbpp_dev"], parallel=PAR)) as ex:
    tok = tokenizer()
    out = {}
    for prec in ["f16", "q4_0"]:
        with LlamaServer(GGUF[prec], parallel=PAR, log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            for s in ["humanevalfix", "mbpp_dev"]:
                summ, rows = repair_loop(srv, tok, load_suite(s), max_turns=3)
                out[f"{prec}/{s}"] = summ
                with open(os.path.join(ex.dir, f"outputs_{prec}_{s}.jsonl"), "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                ex.log(prec, s, summ)
                ex.result(**out)
    ex.result(**out, headline=" | ".join(f"{k}: {v['pass_rate']:.3f} (turns {v['mean_turns']}, false-conf "
                                         f"{v['false_confidence']})" for k, v in out.items()))

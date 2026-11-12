"""EXP-029: official Evaluator on real competition tasks, on this machine: harness in the Linux container (subprocess
sandbox, as in the organisers' notebook), llama-server on the Arc GPU, E2B at Q4_0 and F16.

Agent: the sample submission with the E2B model and no adapters (graph tools, thinking budget, sampling as shipped).
Budgets: harness defaults (60 minutes, 100 tool calls, 300 s commands, 32768 token context). No artificial limits.
Tasks: small-fix tasks that passed the EXP-028 controls; seeded split, first 25 are dev.
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, sweagent as A  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF  # noqa: E402

SLOTS = 4
E28 = [d for d in os.listdir(os.path.join(ROOT, "experiments")) if d.startswith("EXP-028_")][0]
rows = json.load(open(os.path.join(ROOT, "experiments", E28, "controls.json")))
sound = sorted(r["id"] for r in rows if r["sound"])
random.Random(20261106).shuffle(sound)
DEV = sound[:2] if SMOKE else sound[:25]

with Experiment("EXP-029", "official-baseline-e2b-local", category="baseline", depends_on=["EXP-028", "EXP-003"],
                motivation="The true E2B baseline: real issues, the official Evaluator, no artificial budget, both "
                           "precisions, on the author's hardware. It also reveals real per-task cost here.",
                hypothesis="E2B resolves more than the 5-minute Kaggle run's 6.9% once limits are lifted, roughly 10-15% "
                           "of small-fix dev tasks. Q4_0 and F16 differ by no more than two tasks.",
                config=dict(tasks=DEV, slots=SLOTS, ctx_per_slot=32768, max_tool_calls=100, max_time_minutes=60,
                            n_sound=len(sound), submission="sample_submission with E2B, no adapters")) as ex:
    A.ensure_docker(ex.log)
    out = {}
    rel = os.path.relpath(ex.dir, ROOT).replace(os.sep, "/")
    idp = os.path.join(ex.dir, "ids.json")
    json.dump(DEV, open(idp, "w"))
    for prec in ["q4_0", "f16"]:
        rd = f"{rel}/results_{prec}"
        with LlamaServer(GGUF[prec], parallel=SLOTS, ctx_per_slot=32768, extra_args=("--host", "0.0.0.0"),
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")):
            p = subprocess.run(["docker", "run", "--rm", "-v", f"{ROOT}:/proj", "-w", "/proj",
                                "--add-host", "host.docker.internal:host-gateway",
                                "-e", "LOCAL_INFERENCE_URL=http://host.docker.internal:8091/v1", "-e", "LOCAL_API_KEY=EMPTY",
                                "-e", "OPENAI_API_KEY=EMPTY", "-e", "PYTHONIOENCODING=utf-8", "-e", "LITELLM_LOCAL_MODEL_COST_MAP=True",
                                "-e", "OTEL_SDK_DISABLED=true", "-e", "LITELLM_LOG=WARNING", "sne-harness:latest", "python", "-u",
                                "tools/official_agent.py", rd, f"{rel}/ids.json", str(SLOTS), "60", "100"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            ex.log(p.stdout[-2500:])
            ex.log(p.stderr[-2500:])
        if "RUN_DONE" not in p.stdout:
            raise RuntimeError(f"harness did not finish for {prec}: rc={p.returncode}")
        sp = os.path.join(ROOT, rd, "summary.json")
        out[prec] = json.load(open(sp))
        tr = [json.loads(l) for l in open(os.path.join(ROOT, rd, "task_results.jsonl"), encoding="utf-8")]
        calls = sum(r.get("total_llm_calls") or 0 for r in tr)
        tools = sum(r.get("tool_calls") or 0 for r in tr)
        ex.log(prec, "llm calls", calls, "tool calls", tools)
        if calls == 0 or tools == 0:  # QA: a run where the model never acted is a failure, whatever the exit status
            raise RuntimeError(f"no model activity for {prec}: {str(tr[0].get('error'))[:300]}")
        ex.log(prec, {k: v for k, v in out[prec].items() if not isinstance(v, (list, dict))})
        ex.result(**out)
    ex.result(**out, headline=" | ".join(f"{k}: {v.get('resolved')}/{v.get('total_tasks')}" for k, v in out.items()))

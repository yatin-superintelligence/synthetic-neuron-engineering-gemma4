"""EXP-158 (D-216): EXP-154's neuron scan on Gemma 4 12B. The scan is a tool: experiments/scripts/exp154_31b_success_neurons.py with
SCAN_MODEL, SCAN_BENCH and SCAN_ID set; this file only sets them and runs it. Bench: artifacts_local/EXP-158/success_bench.jsonl.gz
(tools/analysis/build_success_bench.py --model 12b), the first edits to a reference-fix file in 12B's own logged runs of the tasks it
solves only sometimes. Outputs: artifacts_local/EXP-158/{fit.json, neurons.pt}."""
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("SCAN_MODEL", "google/gemma-4-12B-it-qat-q4_0-unquantized")
os.environ.setdefault("SCAN_BENCH", os.path.join(ROOT, "artifacts_local", "EXP-158", "success_bench.jsonl.gz"))
os.environ.setdefault("SCAN_ID", "EXP-158")
os.environ.setdefault("SCAN_SLUG", "12b-success-neurons")
runpy.run_path(os.path.join(HERE, "exp154_31b_success_neurons.py"), run_name="__main__")

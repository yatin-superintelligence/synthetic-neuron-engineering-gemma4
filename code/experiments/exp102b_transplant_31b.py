"""EXP-102b: EXP-102's neuron transplant with Gemma 4 31B (QAT, bf16) as the donor. Same method, layers, arms and controls (D-093)."""
import os
import runpy

os.environ["DONOR"] = "31B"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp102_neuron_transplant.py"), run_name="__main__")

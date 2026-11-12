"""EXP-162b (D-225): EXP-162 on all 38 pairs, every prompt longer than 5,120 tokens cut to its first 1,024 and last 4,096 tokens
before the pass (EXP-162 fitted 6 of 38: memory grows about 4.5 MB per prompt token). Runs exp162_31b_causal_dose.py with
EXP162_ID=EXP-162b, EXP162_CUT_ALL=1, EXP162_TAIL=4096 (upload that script with --data)."""
import os
import runpy

os.environ.update(EXP162_ID="EXP-162b", EXP162_CUT_ALL="1", EXP162_TAIL="4096")
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp162_31b_causal_dose.py"), run_name="__main__")

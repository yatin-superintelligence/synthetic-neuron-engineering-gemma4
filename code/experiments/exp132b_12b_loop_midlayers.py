"""EXP-132b launcher (D-158a): EXP-132's script with the edit placed on layers 23, 24, 25 and 26, the four layers where the loop
direction's held-out d' is at least 1.0 and highest relative to the largest of 32 random directions (2.19, 3.15, 2.62 and 1.65 times;
EXP-132's layers.json). EXP-132's own rule, the top 4 by d', picked layers 33, 38, 39 and 40, where d' is the same within noise (1.05
to 1.10 across layers 23 to 40) but random directions separate almost as well (1.13 to 1.46 times). colab_job cannot pass environment
variables, so this wrapper sets them and runs the original script unchanged."""
import os
import runpy

os.environ["EXP132_ID"] = "EXP-132b"
os.environ["EXP132_LAYERS"] = "23,24,25,26"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp132_12b_loop_direction.py"), run_name="__main__")

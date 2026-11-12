"""EXP-109e: EXP-109's total-effect selection with E4B as donor (an L4 suffices); does choosing by total effect beat the direct-logit choice?"""
import os
import runpy

os.environ["DONOR"] = "E4B"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp109_transplant_total_effect.py"), run_name="__main__")

import os, runpy
os.environ["HALF"] = "dev"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp110_success_distillation.py"), run_name="__main__")

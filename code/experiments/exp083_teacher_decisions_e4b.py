"""EXP-083: what would Gemma 4 31B do at E4B's own agent states? Teacher labels for E4B's synthetic decision neurons (EXP-084).

The same procedure as EXP-079 (experiments/scripts/exp079_teacher_decisions.py, whose docstring fixes every choice), applied to the
states of two unedited E4B official runs instead of E2B's three: EXP-073a (seed 2) and EXP-073c (seed 4), both made on a Colab
L4: 1,751 states (863 on dev-split tasks, 888 on test-split tasks), the same scale as EXP-079's 1,745. All three E4B runs would give
2,605 states and about 2.5 h of G4 time; two keep the cost at EXP-079's (about 17 units). Why E4B: it solves about twice as many tasks as E2B (11 and 12 of 58 at those two seeds, 6 at seed 1
on Kaggle, against 4 to 6 for E2B), so a behavioural change that does not raise E2B's count (EXP-080) may raise E4B's; and it is the small-model
candidate for the main track. Outputs: artifacts_local/EXP-083/teacher_labels.jsonl and summary.json (ids, actions, token counts).
Usage on a runtime with the official stack and the 31B model: /opt/sne/bin/python exp083_teacher_decisions_e4b.py [--smoke]
"""
import os
import sys

os.environ.setdefault("SNE_TEACHER_RUNS", "exp-073a,exp-073c")
os.environ.setdefault("SNE_TEACHER_OUT", "EXP-083")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp079_teacher_decisions as T  # noqa: E402

if __name__ == "__main__":
    T.main()

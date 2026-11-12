"""EXP-016: prompt-only baselines. Five system prompts vs none, on the DEV sets only (MBPP+ dev, RepoPatch dev).

Why: (1) every weight edit must beat what a prompt achieves for free; (2) the best prompt is the source behaviour
for prompt-to-neuron compilation (EXP-058), which only makes sense if some prompt actually helps.
Selection rule (pre-registered): the prompt with the highest mean of the two dev pass rates at F16 is 'best';
ties broken by Q4_0 mean.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402

PROMPTS = {
    "expert": "You are an expert Python programmer. Write correct, efficient code.",
    "careful": ("You are a careful software engineer. Read the task closely, consider edge cases such as empty "
                "inputs, negative numbers and duplicates, and make sure your code matches the examples exactly."),
    "stepwise": "Before writing code, briefly reason about the problem and its edge cases. Then give the solution.",
    "terse": "Respond with code only. Do not explain.",
    "agent": ("You are an autonomous coding agent. You fix problems by making minimal, precise changes, and you check "
              "your changes against the tests and the specification before answering."),
}
SUITES = ["mbpp_dev", "repopatch_dev"]
PAR = int(os.environ.get("SNE_PARALLEL", "8"))

with Experiment("EXP-016", "system-prompt-baselines", category="baseline", depends_on=["EXP-010"],
                motivation="Prompt-only baseline for all edits, and source behaviour for prompt-to-neuron compilation.",
                hypothesis="At least one prompt changes dev pass rate by >= 2 points; 'terse' helps format compliance on "
                           "RepoPatch, 'stepwise' lengthens outputs and may truncate.",
                config=dict(prompts=PROMPTS, suites=SUITES, parallel=PAR,
                            selection_rule="max mean dev pass rate at F16, tie-break Q4_0")) as ex:
    arms = []
    for prec in ["f16", "q4_0"]:
        arms.append(Arm(f"base_{prec}", prec))
        arms += [Arm(f"{k}_{prec}", prec, system=v) for k, v in PROMPTS.items()]
    res, comps = evaluate_arms(ex, arms, SUITES, reference="base", parallel=PAR)
    score = {k: (sum(res[f"{k}_f16"][s]["pass_rate"] for s in SUITES) / 2,
                 sum(res[f"{k}_q4_0"][s]["pass_rate"] for s in SUITES) / 2) for k in ["base"] + list(PROMPTS)}
    best = max(PROMPTS, key=lambda k: score[k])
    ex.result(dev_scores=score, best_prompt=best, best_prompt_text=PROMPTS[best],
              headline=f"best={best} f16 {score[best][0]:.3f} vs base {score['base'][0]:.3f}; q4 {score[best][1]:.3f} "
                       f"vs {score['base'][1]:.3f}")

"""EXP-061: what else does the loop edit change? Everyday behaviour and general ability.

Required before any claim. Suites: the 40 fixed conversational probes, MBPP+ dev for code, and MMLU-500 as a general control, at both
precisions, base against edit, greedy as in EXP-010. Short prompts, so 8 slots are right here (EXP-001). Responses are saved so
the probes can be read side by side, which is a standing rule after any weight edit.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne.artifacts import LOCAL  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import Arm, evaluate_arms  # noqa: E402

SUITES = ["behavior", "mbpp_dev", "mmlu500"]
A = {p: os.path.join(LOCAL, "EXP-059", f"remove_loop_dirs_{p}.gguf") for p in ("f16", "q4_0")}

with Experiment("EXP-061", "loop-edit-side-effects", category="improvement", depends_on=["EXP-059"],
                motivation="An edit that removes loops may also damage ordinary repetition, lists, code or recall.",
                hypothesis="Scores stay within the noise floor of EXP-004 and the probes read normally.",
                config=dict(suites=SUITES, adapter="EXP-059 remove_loop_dirs")) as ex:
    arms = [Arm("base_f16", "f16"), Arm("edit_f16", "f16", lora=A["f16"]), Arm("base_q4_0", "q4_0"), Arm("edit_q4_0", "q4_0", lora=A["q4_0"])]
    res, comps = evaluate_arms(ex, arms, SUITES, reference="base")
    table = {a.name: {s: res[a.name][s].get("pass_rate") for s in SUITES} for a in arms}
    ex.save_json("table.json", table)
    ex.result(table=table, comparisons=comps, headline="; ".join(f"{a.name} {table[a.name]}" for a in arms))

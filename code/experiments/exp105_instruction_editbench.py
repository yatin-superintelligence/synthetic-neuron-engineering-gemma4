"""EXP-105: the lab's best instruction ('all3', D-098, D-102) on real edit states from the official traces (EditBench), at F16.

The instruction raised applying edits on the injected-bug bench (held 257 states: 94 against 74 to 76 for three no-edit arms). This
run asks whether it does so on the states of real official runs: EditBench replays the agent state before each recorded edit_file
call of the official traces and scores the reply against the file as it stood at that step (sne.editbench). Same states, seed and
server settings as EXP-097b and EXP-099b (prompts up to 12,000 tokens, one state per task and run, seed 0, official sampling), so their
unedited arm is reused. Arms: unedited; the instruction appended to the system message; a neutral instruction of about the same
length (EXP-104's control text). Outcomes: applied edits, edits whose old_string carries a backslash, not-found failures; exact
McNemar by state, instruction against unedited and against the neutral text. Pre-registered in D-103.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, editbench as E, loops as L  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402
from sne.stats import mcnemar_exact  # noqa: E402

INSTR = ("The bug is one small mistake on one line. Once you have read the code that contains it, fix it right away with edit_file. "
         "Copy old_string character for character from the file content you read, without adding escape characters, and in new_string "
         "change only the wrong token or operator. Do not rewrite or restructure code.")
CTRL = ("The repository is a Python project. Its files use four spaces for indentation and its tests are written with pytest. "
        "The tools listed above are the only ones available in this session, and every tool call returns its result as JSON. "
        "Paths are given relative to the repository root, which is also the working directory of every command you run.")
MAX_PROMPT = 12000
TEXTS = {"instruction": INSTR, "neutral": CTRL}

with Experiment("EXP-105", "instruction-editbench-f16", category="improvement", depends_on=["EXP-097b", "EXP-099b"],
                motivation="The one bench result outside the noise floor is an instruction (D-102); does it raise applied edits on real "
                           "official-run states?",
                hypothesis="On F16 EditBench the instruction raises applied edit calls and lowers backslash-carrying ones against the unedited "
                           "model and against a neutral instruction of the same length (exact McNemar by state).",
                config=dict(max_prompt=MAX_PROMPT, per_run_task=1, seed=0, precision="f16", texts=TEXTS)) as ex:
    tok = tokenizer()
    cases = E.short_cases(tok, E.load(None, per_run_task=1), MAX_PROMPT)[:4 if SMOKE else None]
    ex.log("states", len(cases), "tasks", len({c["instance_id"] for c in cases}), "originally applied", sum(c["ok"] for c in cases))
    out, rows = {}, {}
    prev = glob.glob(os.path.join(ROOT, "experiments", "EXP-097b_*", "rows_base.json"))
    if prev and not SMOKE:
        rws = json.load(open(prev[0], encoding="utf-8"))
        if {r["case"] for r in rws} == {c["case"] for c in cases}:
            out["base"] = json.load(open(os.path.join(os.path.dirname(prev[0]), "results.json"), encoding="utf-8")).get("results", {}).get("arms", {}).get("base")
            rows["base"] = {r["case"]: r for r in rws}
            ex.log("base arm reused from", prev[0])
    with LlamaServer(GGUF["f16"], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                     log_path=os.path.join(ex.dir, "server_f16.log")) as srv:
        for name in (["base"] if "base" not in rows else []) + list(TEXTS):
            fn = None if name == "base" else (lambda t, c, s=chr(10) * 2 + TEXTS[name]: E.with_system_suffix(t, c, s))
            summ, rws = E.run(srv, tok, cases, seeds=(0,), prompt_fn=fn)
            out[name], rows[name] = summ, {r["case"]: r for r in rws}
            ex.save_json(f"rows_{name}.json", rws)
            ex.log(name, summ["all"], summ["mean_tokens"])
            ex.result(arms=out)
    common = sorted(set.intersection(*(set(v) for v in rows.values())))

    def flag(name, key):
        return [bool(rows[name][c]["tool"] == "edit_file" and (rows[name][c][key] if key == "backslash" else rows[name][c]["outcome"] == "applied"))
                for c in common]
    tests = {}
    for other in ("base", "neutral"):
        for key in ("backslash", "applied"):
            a, b = flag("instruction", key), flag(other, key)
            x = sum(1 for i, j in zip(a, b) if i and not j)
            y = sum(1 for i, j in zip(a, b) if j and not i)
            tests[f"instruction_vs_{other}_{key}"] = dict(instruction_only=x, other_only=y, p=round(mcnemar_exact(x, y), 4))
    tot = {n: dict(backslash=sum(flag(n, "backslash")), applied=sum(flag(n, "applied")), edit_calls=out[n]["all"]["edit_calls"]) for n in rows}
    ex.result(arms=out, states=len(common), totals=tot, tests=tests,
              headline=f"F16 EditBench ({len(common)} states): applied unedited {tot['base']['applied']}, instruction {tot['instruction']['applied']}, "
                       f"neutral {tot['neutral']['applied']}; backslash {tot['base']['backslash']}, {tot['instruction']['backslash']}, {tot['neutral']['backslash']}")

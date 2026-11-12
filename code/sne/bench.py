"""Benchmarks: prompt construction, code extraction, program assembly, and fixed, seeded splits.

Split policy (fixed before any intervention is evaluated, to prevent selection leakage):
  * Neuron/direction SELECTION data : MBPP (original, full) train split solutions + prose (see probes.py).
  * Hyper-parameter DEV set        : MBPP+ problems whose task_id is in the dev half (seeded split).
  * Held-out TEST sets              : MBPP+ test half, HumanEval+ (all 164), HumanEvalFix-Python (all 164).
"""
import json
import os
import random
import re

from . import DATA

SPLIT_SEED = 20261105


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ----------------------------------------------------------------------------- extraction
FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def extract_code(text, entry_point=None):
    """Return the most plausible Python code block from a chat response."""
    blocks = FENCE.findall(text)
    if not blocks:  # unterminated fence (e.g. truncation) -> take everything after the opening fence
        m = re.search(r"```(?:python|py)?[ \t]*\n(.*)", text, re.S)
        blocks = [m.group(1)] if m else [text]
    if entry_point:
        hits = [b for b in blocks if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b)]
        if hits:
            return max(hits, key=len)
    return max(blocks, key=len)


# ----------------------------------------------------------------------------- task records
SYSTEM_NOTE = ""  # no system prompt by default (Gemma chat template); prompt-baselines set this explicitly


def humanevalplus():
    out = []
    for r in _load("humanevalplus.jsonl"):
        user = ("Complete the following Python function. Respond with the complete function "
                "(including the signature) in a single ```python code block.\n\n"
                f"```python\n{r['prompt']}```")
        out.append(dict(bench="humaneval+", task_id=r["task_id"], user=user, entry_point=r["entry_point"],
                        _prompt=r["prompt"], _test=r["test"], _canonical=r["prompt"] + r["canonical_solution"]))
    return out


def humaneval_program(task, code):
    # Prompt (helpers + docstring-only stub) first, then model code re-defines the target function.
    return (task["_prompt"] + "\n    pass\n\n" + code + "\n\n" + task["_test"]
            + f"\n\ncheck({task['entry_point']})\n")


def mbppplus(split=None):
    rows = _load("mbppplus.jsonl")
    ids = sorted(r["task_id"] for r in rows)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ids)
    dev = set(ids[: len(ids) // 2])
    out = []
    for r in rows:
        s = "dev" if r["task_id"] in dev else "test"
        if split and s != split:
            continue
        m = re.search(r"assert\s+(?:set\()?\s*(?:not\s+)?\(?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", r["test_list"][0])
        ep = m.group(1) if m else None
        user = (f"{r['prompt']}\nYour code should pass this test:\n{r['test_list'][0]}\n\n"
                "Respond with the complete Python solution in a single ```python code block.")
        out.append(dict(bench="mbpp+", split=s, task_id=f"Mbpp/{r['task_id']}", user=user, entry_point=ep,
                        _imports="\n".join(r["test_imports"]), _test=r["test"], _canonical=r["code"],
                        _visible=r["test_list"][0]))
    return out


def mbpp_program(task, code, visible=False):
    test = task["_visible"] if visible else task["_test"]
    return task["_imports"] + "\n" + code + "\n\n" + test + "\n"


def humanevalfix():
    out = []
    for r in _load("humanevalfix_python.jsonl"):
        buggy = r["declaration"] + r["buggy_solution"]
        user = (f"The following Python function `{r['entry_point']}` contains a bug.\n\n"
                f"Specification:\n{r['docstring'].strip()}\n\n"
                f"```python\n{r['import']}{buggy}```\n\n"
                f"These example tests must pass:\n```python\n{r['example_test'].strip()}\n```\n\n"
                "Fix the bug. Respond with the complete corrected function in a single ```python code block.")
        out.append(dict(bench="humanevalfix", task_id=r["task_id"], user=user, entry_point=r["entry_point"],
                        _import=r["import"] + r.get("test_setup", ""), _test=r["test"],
                        _example_test=r["example_test"], _buggy=buggy, bug_type=r["bug_type"],
                        _canonical=r["declaration"] + r["canonical_solution"]))
    return out


def fix_program(task, code, visible=False):
    test = task["_example_test"] if visible else task["_test"]
    return task["_import"] + "\n" + code + "\n\n" + test + "\n"


def build_program(task, code, visible=False):
    b = task["bench"]
    if b == "humaneval+":
        return humaneval_program(task, code)
    if b == "mbpp+":
        return mbpp_program(task, code, visible=visible)
    if b == "humanevalfix":
        return fix_program(task, code, visible=visible)
    raise ValueError(b)


# Tasks whose reference solution cannot pass in this environment (found by EXP-002); excluded from all scoring.
EXCLUDE = {
    "Mbpp/590": "expected output computed on Linux libm; canonical solution differs in last float digit on Windows",
    "HumanEval/32": "HF evalplus test harness bug: `_poly(*candidate(*inp), inp)` unpacks a float (TypeError)",
}


def get_suite(name):
    tasks = {
        "mbpp_dev": lambda: mbppplus("dev"),
        "mbpp_test": lambda: mbppplus("test"),
        "humaneval": humanevalplus,
        "humanevalfix": humanevalfix,
    }[name]()
    return [t for t in tasks if t["task_id"] not in EXCLUDE]


# ----------------------------------------------------------------------------- MMLU (general-capability control)
def mmlu_sample(n=500, seed=SPLIT_SEED):
    rows = _load("mmlu_all_test.jsonl")
    rng = random.Random(seed)
    by_subj = {}
    for i, r in enumerate(rows):
        by_subj.setdefault(r["subject"], []).append(i)
    subjects = sorted(by_subj)
    picked = []
    k = 0
    while len(picked) < n:  # round-robin stratified sample over subjects
        s = subjects[k % len(subjects)]
        pool = by_subj[s]
        if pool:
            picked.append(pool.pop(rng.randrange(len(pool))))
        k += 1
    out = []
    for i in picked:
        r = rows[i]
        ch = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
        user = (f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n"
                f"{r['question']}\n{ch}\n\nAnswer with only the letter of the correct option.")
        out.append(dict(bench="mmlu", task_id=f"mmlu/{i}", user=user, answer=r["answer"]))
    return out

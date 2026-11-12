"""LiveCodeBench (Liu et al., 2024) loader and scorer: competitive-programming problems, harder than HumanEval/MBPP.

Data: livecodebench/code_generation_lite test6.jsonl (newest public batch), pinned revision. Private tests are stored
as base64(zlib(pickle(json string))). Pickle from a download can execute code, so a restricted unpickler that resolves
no globals at all is used; anything other than a plain string payload raises.
Scoring: all public + up to MAX_TESTS private tests pass in a single process per task, 60 s wall limit. The cap keeps
programs small; a task can only be easier than under the full suite, so absolute scores are an upper bound relative to
official numbers and should be used for comparisons between arms, with Google's 44.0% only as a coarse calibration.
"""
import base64
import io
import json
import os
import pickle
import zlib

from . import DATA, bench
from .evaluate import render, strip_thought
from .execute import run_programs

REV = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
PATH = os.path.join(DATA, "lcb", "test6.jsonl")
MAX_TESTS = 25


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def decode_private(s):
    data = zlib.decompress(base64.b64decode(s.encode("utf-8")))
    obj = _NoGlobals(io.BytesIO(data)).load()
    if isinstance(obj, bytes):
        obj = obj.decode("utf-8")
    if not isinstance(obj, str):
        raise ValueError("unexpected private test payload")
    return json.loads(obj)


def load():
    tasks = []
    with open(PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pub = json.loads(r["public_test_cases"])
            try:
                priv = decode_private(r["private_test_cases"])
            except Exception:  # noqa: BLE001
                priv = json.loads(r["private_test_cases"])
            meta = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            tests = pub + priv[:MAX_TESTS]
            fn = meta.get("func_name")
            starter = r.get("starter_code") or ""
            if starter.strip():
                user = (f"{r['question_content']}\n\nComplete this code and respond with the full solution in a single "
                        f"```python code block.\n```python\n{starter}\n```")
            else:
                user = (f"{r['question_content']}\n\nWrite a Python program that reads from standard input and writes to "
                        "standard output. Respond with the program in a single ```python code block.")
            tasks.append(dict(bench="lcb", task_id=r["question_id"], user=user, difficulty=r.get("difficulty"),
                              platform=r.get("platform"), date=r.get("contest_date"), func=fn, tests=tests))
    return tasks


DRIVER = r'''
import io, json, sys
_code = {code!r}
_tests = json.loads({tests!r})
_func = {func!r}
_cmp = lambda a, b: a.strip().split() == b.strip().split()
if _func:
    _ns = {{"__name__": "solution"}}
    exec("from typing import *\nimport collections, math, heapq, bisect, itertools, functools\n" + _code, _ns)
    for t in _tests:
        args = [json.loads(x) for x in t["input"].strip().split("\n")]
        exp = json.loads(t["output"])
        got = getattr(_ns["Solution"](), _func)(*args)
        assert got == exp, "wrong answer"
else:
    _comp = compile(_code, "sol", "exec")
    _real = sys.stdout
    for t in _tests:
        sys.stdin = io.StringIO(t["input"])
        sys.stdout = buf = io.StringIO()
        try:
            exec(_comp, {{"__name__": "__main__"}})
        except SystemExit:
            pass
        sys.stdout = _real
        assert _cmp(buf.getvalue(), t["output"]), "wrong answer"
'''


def run(server, tok, tasks, system=None, lora_scales=None, max_tokens=8192, thinking=False):
    prompts = render(tok, [t["user"] for t in tasks], system, thinking)
    gens, st = server.complete_many(prompts, max_tokens=max_tokens, lora_scales=lora_scales)
    progs = []
    for t, g in zip(tasks, gens):
        code = bench.extract_code(strip_thought(g["text"]))
        progs.append((t["task_id"], DRIVER.format(code=code, tests=json.dumps(t["tests"]), func=t["func"])))
    res = run_programs(progs, timeout=60)
    rows = [dict(task_id=t["task_id"], passed=res[t["task_id"]]["passed"], status=res[t["task_id"]]["status"],
                 difficulty=t["difficulty"], stop_type=g["stop_type"], n_gen=g["n_gen"], response=g["text"],
                 stderr_tail=res[t["task_id"]]["stderr_tail"][-300:]) for t, g in zip(tasks, gens)]
    by = {}
    for d in ("easy", "medium", "hard"):
        v = [r for r in rows if r["difficulty"] == d]
        by[d] = dict(n=len(v), rate=round(sum(r["passed"] for r in v) / max(1, len(v)), 4))
    n = len(rows)
    return dict(bench="lcb", n=n, passed=sum(r["passed"] for r in rows), pass_rate=round(sum(r["passed"] for r in rows) / n, 4),
                by_difficulty=by, truncated=sum(r["stop_type"] == "limit" for r in rows), gen=st), rows

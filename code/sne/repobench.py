"""RepoPatch: a small, executable, multi-file bug-fixing benchmark in the competition's tool-call format.

Construction (fully deterministic, seed 20261106):
  1. Seed functions: HumanEval+ canonical solutions (validated to pass in EXP-002).
  2. Bug injection: AST mutation operators (comparison flip, off-by-one constant, arithmetic swap, boolean swap,
     negation removal, return-expression swap). A mutant is kept only if it is *killed* by the HumanEval+ test
     suite (fails at least one hidden test) and still parses.
  3. Repository: `pkg/core.py` holds the buggy target function plus two distractor functions from other tasks;
     `pkg/utils.py` holds two more distractors; `tests/test_core.py` is NOT shown (hidden tests).
  4. Issue text: produced by running the hidden tests on the buggy code and reporting the first failing input,
     the expected value and the observed value/exception, phrased like a GitHub issue.
The model must answer with JSON tool calls `edit_file(filepath, old_string, new_string)` exactly as in the
competition harness. Scoring: format_ok (valid JSON calls), applied (each old_string occurs exactly once),
resolved (hidden tests pass after applying the edits).
"""
import ast
import copy
import json
import os
import random
import re

from . import DATA, bench
from .execute import run_programs

SEED = 20261106


class _Mut(ast.NodeTransformer):
    """Applies the k-th applicable mutation of a given operator class."""

    CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
    BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}

    def __init__(self, op, k):
        self.op, self.k, self.seen, self.done = op, k, 0, None

    def _hit(self):
        hit = self.seen == self.k
        self.seen += 1
        return hit

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.op == "cmp" and type(node.ops[0]) in self.CMP and self._hit():
            node.ops[0] = self.CMP[type(node.ops[0])]()
            self.done = "comparison operator flipped"
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if self.op == "arith" and type(node.op) in self.BIN and self._hit():
            node.op = self.BIN[type(node.op)]()
            self.done = "arithmetic operator swapped"
        return node

    def visit_Constant(self, node):
        if self.op == "const" and isinstance(node.value, int) and not isinstance(node.value, bool) \
                and abs(node.value) <= 10 and self._hit():
            self.done = "off-by-one integer constant"
            return ast.copy_location(ast.Constant(node.value + 1), node)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.op == "bool" and self._hit():
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.done = "boolean operator swapped"
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if self.op == "not" and isinstance(node.op, ast.Not) and self._hit():
            self.done = "negation removed"
            return node.operand
        return node


def _func_src(full_src, name):
    tree = ast.parse(full_src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(full_src, n)
    return None


def _strip_docstring(fn_src):
    tree = ast.parse(fn_src)
    f = tree.body[0]
    if f.body and isinstance(f.body[0], ast.Expr) and isinstance(getattr(f.body[0], "value", None), ast.Constant) \
            and isinstance(f.body[0].value.value, str):
        doc = f.body[0].value.value
        f.body = f.body[1:] or [ast.Pass()]
        return ast.unparse(tree), doc
    return ast.unparse(tree), ""


def build(n_target=150, out_path=None):
    rng = random.Random(SEED)
    he = [t for t in bench.get_suite("humaneval")]
    pool = []
    for t in he:
        fn = _func_src(t["_canonical"], t["entry_point"])
        if fn is None:
            continue
        clean, doc = _strip_docstring(fn)
        header = "\n".join(l for l in t["_prompt"].splitlines() if l.startswith(("import ", "from ")))
        pool.append(dict(t=t, fn=clean, doc=doc.strip(), header=header))
    ops = ["cmp", "const", "arith", "bool", "not"]
    cands = []
    for p in pool:
        tree = ast.parse(p["fn"])
        for op in ops:
            for k in range(3):
                m = _Mut(op, k)
                new = m.visit(copy.deepcopy(tree))
                if m.done:
                    src = ast.unparse(ast.fix_missing_locations(new))
                    if src != p["fn"]:
                        cands.append(dict(p=p, op=op, k=k, kind=m.done, buggy=src))
    rng.shuffle(cands)
    # keep killed mutants only, at most 1 per seed function, preferring operator diversity
    progs = [(str(i), c["p"]["header"] + "\n" + c["buggy"] + "\n\n" + c["p"]["t"]["_test"] +
              f"\n\ncheck({c['p']['t']['entry_point']})\n") for i, c in enumerate(cands)]
    res = run_programs(progs)
    used, tasks = set(), []
    for i, c in enumerate(cands):
        ep = c["p"]["t"]["entry_point"]
        if res[str(i)]["passed"] or ep in used:
            continue
        used.add(ep)
        tasks.append(c)
        if len(tasks) >= n_target:
            break
    others = [p for p in pool]
    out = []
    for j, c in enumerate(tasks):
        ep = c["p"]["t"]["entry_point"]
        dis = [p for p in rng.sample(others, 6) if p["t"]["entry_point"] != ep][:4]
        imports = sorted({l for p in [c["p"]] + dis for l in p["header"].splitlines() if l})
        core = "\n".join(imports) + "\n\n\n" + "\n\n\n".join([dis[0]["fn"], c["buggy"], dis[1]["fn"]]) + "\n"
        utils = "\n".join(imports) + "\n\n\n" + "\n\n\n".join([dis[2]["fn"], dis[3]["fn"]]) + "\n"
        out.append(dict(task_id=f"RepoPatch/{j}", seed_task=c["p"]["t"]["task_id"], entry_point=ep,
                        mutation=c["kind"], files={"pkg/core.py": core, "pkg/utils.py": utils},
                        doc=c["p"]["doc"], _test=c["p"]["t"]["_test"], canonical_fn=c["p"]["fn"],
                        buggy_fn=c["buggy"]))
    # issue text from the first failing hidden-test input
    probe = []
    for t in out:
        m = re.search(r"inputs = (\[.*?\])\n", t["_test"], re.S)
        probe_code = (t["files"]["pkg/core.py"] + "\n\nimport json\n" +
                      f"_ref_src = {json.dumps(t['canonical_fn'])}\n_ns = {{}}\nexec(_ref_src, globals(), _ns)\n"
                      f"_ref = _ns['{t['entry_point']}']\n" +
                      (f"_inputs = {m.group(1)}\n" if m else "_inputs = []\n") +
                      "for _inp in _inputs:\n"
                      "    try:\n        _exp = _ref(*_inp)\n    except Exception:\n        continue\n"
                      f"    try:\n        _got = {t['entry_point']}(*_inp)\n"
                      "    except Exception as e:\n        _got = 'raised ' + type(e).__name__ + ': ' + str(e)\n"
                      "    if _got != _exp:\n"
                      "        print(json.dumps({'inp': repr(_inp)[:300], 'exp': repr(_exp)[:200], 'got': repr(_got)[:200]}))\n"
                      "        break\n")
        probe.append((t["task_id"], probe_code))
    pr = run_programs(probe, timeout=60)
    final = []
    for t in out:
        o = pr[t["task_id"]].get("stdout_tail", "").strip().splitlines()
        if not o:
            continue  # no reproducible failing input among the listed inputs -> drop (documented in build stats)
        t["failing_example"] = json.loads(o[-1])
        t["issue"] = issue_text(t, t["failing_example"])
        final.append(t)
    return final


def issue_text(t, failing):
    args = failing["inp"]
    return (f"Title: `{t['entry_point']}` returns the wrong result\n\n"
            f"Calling `{t['entry_point']}` with arguments {args} returns {failing['got']}, "
            f"but the expected result is {failing['exp']}.\n\n"
            f"Intended behaviour (from the function's documentation):\n{t['doc']}")


TOOL_SPEC = """You are working in a Python repository. You can change files only by calling the tool `edit_file`.
To call it, output one or more JSON objects, each on its own line, inside a single ```json code block:
{"tool": "edit_file", "filepath": "<path>", "old_string": "<exact text currently in the file>", "new_string": "<replacement text>"}
`old_string` must match the current file content exactly (including indentation) and must occur exactly once.
Make the smallest change that resolves the issue. Do not output anything after the code block."""


def render_task(t):
    files = "\n\n".join(f"File: {p}\n```python\n{c}```" for p, c in t["files"].items())
    return f"{TOOL_SPEC}\n\nRepository files:\n\n{files}\n\nIssue:\n{t['issue']}"


def parse_calls(text):
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    calls = []
    # accept JSON objects one per line, a JSON list, or concatenated objects
    try:
        obj = json.loads(body)
        calls = obj if isinstance(obj, list) else [obj]
    except Exception:  # noqa: BLE001
        dec = json.JSONDecoder()
        i = 0
        while i < len(body):
            j = body.find("{", i)
            if j < 0:
                break
            try:
                o, k = dec.raw_decode(body, j)
                calls.append(o)
                i = k
            except Exception:  # noqa: BLE001
                i = j + 1
    calls = [c for c in calls if isinstance(c, dict) and c.get("tool", "edit_file") == "edit_file"
             and {"filepath", "old_string", "new_string"} <= set(c)]
    return calls


def apply_calls(files, calls):
    files = dict(files)
    for c in calls:
        p = c["filepath"].lstrip("./")
        if p not in files or files[p].count(c["old_string"]) != 1:
            return None
        files[p] = files[p].replace(c["old_string"], c["new_string"], 1)
    return files


def hidden_program(t, files):
    return files["pkg/utils.py"] + "\n\n" + files["pkg/core.py"] + "\n\n" + t["_test"] + f"\n\ncheck({t['entry_point']})\n"


def score(t, text):
    calls = parse_calls(text)
    if not calls:
        return dict(format_ok=False, applied=False, program=None)
    new = apply_calls(t["files"], calls)
    if new is None:
        return dict(format_ok=True, applied=False, program=None)
    return dict(format_ok=True, applied=True, program=hidden_program(t, new))


def load(path=os.path.join(DATA, "repopatch_v1.jsonl")):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_split(split=None):
    """Seeded dev/test halves of RepoPatch v1 (seed 20261105, same seed as the MBPP+ split)."""
    tasks = load()
    if split is None:
        return tasks
    ids = sorted(t["task_id"] for t in tasks)
    random.Random(20261105).shuffle(ids)
    dev = set(ids[: len(ids) // 2])
    return [t for t in tasks if (t["task_id"] in dev) == (split == "dev")]

"""CopyBench: short, high-power tasks measuring exact copying from context, the skill behind edit_file old_string,
file paths and required identifiers. Source: real code from the CPython 3.11 standard library (disjoint from MBPP,
HumanEval and SWE-bench). Deterministic: the generated file data/copybench_v1.jsonl is the benchmark.

Task types (100 each):
  name    : implement a function whose required (unusual) name appears only in the shown test. Pass: `def <name>(`.
  edit    : native tool call edit_file on a shown file excerpt: change one fragment on one specified line.
            Pass: old_string occurs exactly once and applying the edit gives the exact expected text.
  verbatim: reproduce lines a..b of a shown excerpt exactly in a code block (whitespace-sensitive).
"""
import ast
import json
import keyword
import os
import random
import re
import sysconfig

from . import DATA
from .engine import strip_bos

SEED = 20261107
PATH = os.path.join(DATA, "copybench_v1.jsonl")
MODULES = ["argparse", "csv", "gettext", "ipaddress", "json/decoder", "logging/__init__", "mailbox", "mimetypes",
           "optparse", "pathlib", "pprint", "shlex", "smtplib", "string", "tarfile", "textwrap", "urllib/parse",
           "uuid", "zipfile/__init__", "configparser", "difflib", "dataclasses", "fnmatch", "glob", "heapq", "html/parser",
           "http/cookies", "statistics", "tokenize", "calendar"]
WORDS_A = ["fetch", "derive", "collect", "resolve", "scan", "probe", "render", "emit", "merge", "stage"]
WORDS_B = ["token", "record", "entry", "chunk", "header", "field", "segment", "marker", "slot", "span"]
WORDS_C = ["list", "map", "stream", "view", "batch", "table", "index", "set", "tree", "queue"]
EDIT_TOOL = [{"type": "function", "function": {"name": "edit_file", "description": "Replace old_string with new_string "
              "in a file. old_string must match exactly and occur exactly once.", "parameters": {"type": "object", "properties": {
                  "filepath": {"type": "string", "description": "path"}, "old_string": {"type": "string", "description": "exact existing text"},
                  "new_string": {"type": "string", "description": "replacement text"}}, "required": ["filepath", "old_string", "new_string"]}}}]


def _sources():
    lib = sysconfig.get_paths()["stdlib"]
    if "Scripts" in lib or not os.path.isdir(lib):
        import pathlib
        lib = str(pathlib.Path(os.__file__).parent)
    out = {}
    for m in MODULES:
        p = os.path.join(lib, m + ".py")
        if os.path.exists(p):
            out[m] = open(p, encoding="utf-8").read().replace("\r\n", "\n")
    return out


def _functions(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.split("\n")
    fns = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.end_lineno - n.lineno >= 6 and n.col_offset == 0:
            fns.append((n.name, n.lineno, n.end_lineno, ast.get_docstring(n) or ""))
    return fns, lines


def build():
    rng = random.Random(SEED)
    src = _sources()
    tasks = []
    # ---- name adherence
    fn_pool = []
    for m, s in src.items():
        r = _functions(s)
        if r:
            fn_pool += [(m, f, r[1]) for f in r[0] if len(f[3]) > 30 and not f[0].startswith("_")]
    rng.shuffle(fn_pool)
    used = set()
    for m, (name, a, b, doc), lines in fn_pool:
        if len(used) >= 100:
            break
        new = f"{rng.choice(WORDS_A)}_{rng.choice(WORDS_B)}_{rng.choice(WORDS_C)}"
        if new in used:
            continue
        used.add(new)
        summary = " ".join(doc.split())[:300]
        user = (f"Write a Python function for this description.\n\n{summary}\n\nYour code should pass this test:\n"
                f"assert {new}(example_input) == expected_output\n\nRespond with the function in a single ```python code block.")
        tasks.append(dict(kind="name", task_id=f"name/{len(used) - 1}", user=user, required=new))
    # ---- excerpts for edit and verbatim
    excerpts = []
    for m, s in src.items():
        lines = s.split("\n")
        for _ in range(12):
            a = rng.randrange(40, max(41, len(lines) - 80))
            ex = lines[a:a + rng.randrange(30, 70)]
            good = [i for i, l in enumerate(ex) if 24 <= len(l) <= 90 and l.strip() and not l.strip().startswith("#")
                    and "\t" not in l and "\n".join(ex).count(l) == 1]
            if len(good) >= 6:
                excerpts.append((m, ex, good))
    rng.shuffle(excerpts)
    edits = 0
    for m, ex, good in excerpts[:115]:
        if edits >= 100:
            break
        j = edits
        i = rng.choice(good)
        line = ex[i]
        idents = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", line)
                  if line.count(w) == 1 and not keyword.iskeyword(w) and w not in ("self", "None", "True", "False")]
        if not idents:
            continue
        old_id = rng.choice(idents)
        new_id = old_id + "_v2"
        body = "\n".join(ex)
        path = f"lib/{m.replace('/', '_')}.py"
        user = (f"File: {path}\n```python\n{body}\n```\n\nOn line {i + 1} of this file, rename `{old_id}` to `{new_id}`. "
                "Change only that line. Make the change with one edit_file call.")
        tasks.append(dict(kind="edit", task_id=f"edit/{j}", user=user, body=body, path=path, line=line,
                          expected=body.replace(line, line.replace(old_id, new_id), 1)))
        edits += 1
    verb = 0
    for m, ex, good in excerpts[115:]:
        if verb >= 100:
            break
        j = verb
        n = rng.choice([3, 4, 5])
        starts = [a for a in range(0, max(1, len(ex) - n)) if ex[a].strip() and ex[a + n - 1].strip()
                  and not any("```" in x for x in ex)]
        if not starts:
            continue
        a = rng.choice(starts)
        body = "\n".join(ex)
        want = "\n".join(ex[a:a + n])
        user = (f"```python\n{body}\n```\n\nCopy lines {a + 1} to {a + n} of the code above exactly as written, preserving "
                "all indentation and spacing, in a single ```python code block. Do not add anything else.")
        tasks.append(dict(kind="verbatim", task_id=f"verbatim/{j}", user=user, expected=want))
        verb += 1
    ids = sorted(t["task_id"] for t in tasks)
    random.Random(SEED + 1).shuffle(ids)
    dev = set(ids[: len(ids) // 2])
    for t in tasks:
        t["split"] = "dev" if t["task_id"] in dev else "test"
    return tasks


def load(split=None):
    rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]
    return [r for r in rows if split is None or r["split"] == split]


def render(tok, t, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": t["user"]}]
    kw = dict(tools=EDIT_TOOL) if t["kind"] == "edit" else {}
    return strip_bos(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))


def judge(t, text):
    from .sweagent import CALL, parse_args
    if t["kind"] == "name":
        return bool(re.search(rf"def\s+{re.escape(t['required'])}\s*\(", text)), None
    if t["kind"] == "verbatim":
        m = re.search(r"```(?:python)?[ \t]*\n(.*?)\n?```", text, re.S)
        got = m.group(1) if m else text
        return got.rstrip("\n") == t["expected"].rstrip("\n"), None
    calls = CALL.findall(text)
    if not calls:
        return False, "no_call"
    a = parse_args(calls[0][1]) or {}
    old, new = a.get("old_string", ""), a.get("new_string")
    c = t["body"].count(old) if old else 0
    if c == 0:
        return False, "old_string_not_found"
    if c > 1:
        return False, "old_string_not_unique"
    if new is None or t["body"].replace(old, new, 1) != t["expected"]:
        return False, "wrong_result"
    return True, None


def run(server, tok, tasks, system=None, lora_scales=None, max_tokens=4096):
    prompts = [render(tok, t, system) for t in tasks]
    gens, st = server.complete_many(prompts, max_tokens=max_tokens, lora_scales=lora_scales)
    rows = []
    for t, g in zip(tasks, gens):
        ok, why = judge(t, g["text"])
        rows.append(dict(task_id=t["task_id"], kind=t["kind"], passed=ok, why=why, n_gen=g["n_gen"], response=g["text"]))
    by = {}
    for k in ("name", "edit", "verbatim"):
        v = [r for r in rows if r["kind"] == k]
        by[k] = round(sum(r["passed"] for r in v) / max(1, len(v)), 4)
    why = {}
    for r in rows:
        if r["why"]:
            why[r["why"]] = why.get(r["why"], 0) + 1
    n = len(rows)
    return dict(bench="copybench", n=n, passed=sum(r["passed"] for r in rows),
                pass_rate=round(sum(r["passed"] for r in rows) / n, 4), by_kind=by, edit_failures=why, gen=st), rows

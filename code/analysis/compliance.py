"""Does the agent follow a self-instruction? For each run folder given (artifacts_local/colab/<id>), over every edit_file call:
whether the thought before it opens with (or contains) the quote sentence, quotes code in a fenced block, and whether the edit applied.
Usage: python tools/analysis/compliance.py exp-119p exp-119q exp-114a"""
import glob
from _edits import applied  # noqa: E402
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
KEY = "Before editing, I will copy the exact lines"
for rid in sys.argv[1:]:
    n = opens = contains = fenced = applied = first_n = first_opens = 0
    by = {True: [0, 0], False: [0, 0]}
    for f in glob.glob(os.path.join("artifacts_local", "colab", rid.lower(), "results", "traces", "trace_*.json")):
        t = json.load(open(f, encoding="utf-8"))
        first = True
        for s in t["steps"]:
            if not any(c.get("function_name") == "edit_file" for c in (s.get("tool_calls") or [])):
                continue
            msg = (s.get("message") or "").lstrip()
            o = json.dumps(s.get("observation") or {})
            ok = applied(o)
            n += 1
            opens += msg.startswith(KEY)
            contains += KEY in msg
            fenced += "```" in msg
            applied += ok
            by[KEY in msg][0] += 1
            by[KEY in msg][1] += ok
            if first:
                first_n += 1
                first_opens += KEY in msg
                first = False
    if n:
        print(f"{rid}: edit calls {n}, applied {applied} ({100 * applied / n:.0f}%) | thought opens with the sentence {opens}, contains it {contains}, "
              f"quotes code {fenced} | first edits containing it {first_opens}/{first_n} | applied with sentence {by[True][1]}/{by[True][0]}, without {by[False][1]}/{by[False][0]}")
    else:
        print(rid, "no edit calls found")

"""History compaction and outcome, per task, from the saved traces of an official run and its Kaggle scores.
A compaction is a step whose recorded prompt is at least 3,000 tokens shorter than the previous step's (the official notebook replaces
events older than the last 5 by a model-written summary once the prompt passes 14,336 tokens; the logged prompts of EXP-129a show
that the original problem statement is not in the prompt afterwards). For each task: compactions, whether an edit applied before the
first one, and the official result.
Usage: python tools/analysis/compaction_31b.py [run id, default EXP-074C] [trace folder, default artifacts_local/colab/exp-074c]"""
import ast
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
RUN = (sys.argv[1] if len(sys.argv) > 1 else "EXP-074C").upper()
DIR = sys.argv[2] if len(sys.argv) > 2 else "artifacts_local/colab/exp-074c"
off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"]][x["iid"]] = bool(x["resolved"])
gold = {t for t, v in off["GOLD"].items() if v}


def lit(v):
    try:
        return ast.literal_eval(v) if isinstance(v, str) else (v or {})
    except Exception:
        return {}


rows = []
for f in sorted(glob.glob(os.path.join(DIR, "results", "traces", "trace_*.json"))):
    t = os.path.basename(f)[6:-5]
    prev, comp, calls, edit_before, edits = None, [], 0, False, 0
    for s in json.load(open(f, encoding="utf-8"))["steps"]:
        if s["source"] != "agent":
            continue
        pt = lit(s.get("metrics")).get("prompt_tokens")
        if prev and pt and pt < prev - 3000:
            comp.append(calls)
        prev = pt or prev
        for c in lit(s.get("tool_calls")) or []:
            calls += 1
            ok = '"status": "ok"' in json.dumps(s.get("observation") or "").replace('\\"', '"').replace("'", '"')
            if c.get("function_name") == "edit_file" and ok:
                edits += 1
                if not comp:
                    edit_before = True
    rows.append(dict(t=t, gold=t in gold, solved=bool(off[RUN].get(t)), compactions=len(comp), first_at=comp[0] if comp else None, calls=calls,
                     edit_before=edit_before, edits=edits))
G = [x for x in rows if x["gold"]]
print(f"{RUN}: reference-resolvable tasks {len(G)}, solved {sum(x['solved'] for x in G)}")
for name, f in (("no compaction", lambda x: x["compactions"] == 0), ("1 compaction", lambda x: x["compactions"] == 1),
                ("2 or more compactions", lambda x: x["compactions"] >= 2), ("any compaction", lambda x: x["compactions"] >= 1),
                ("  of those, an edit had applied before the first", lambda x: x["compactions"] >= 1 and x["edit_before"]),
                ("  of those, no edit had applied before the first", lambda x: x["compactions"] >= 1 and not x["edit_before"])):
    a = [x for x in G if f(x)]
    print(f"  {name:52s} tasks {len(a):2d}  solved {sum(x['solved'] for x in a):2d}")
fa = sorted(x["first_at"] for x in G if x["first_at"] is not None)
print("tool calls made before the first compaction: median", fa[len(fa) // 2] if fa else None, "range", (fa[0], fa[-1]) if fa else None)
print("solved after a compaction:", [(x["t"], x["compactions"], x["calls"]) for x in G if x["solved"] and x["compactions"]])
print("unsolved without compaction:", [(x["t"], x["calls"], x["edits"]) for x in G if not x["solved"] and not x["compactions"]])

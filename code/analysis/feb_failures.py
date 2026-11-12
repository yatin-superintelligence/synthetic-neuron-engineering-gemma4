"""Why first edits fail on the first-edit bench (FEB): each recorded first edit_file call, checked against the file as it stood."""
import collections, json, os, sys, difflib
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, "src")
from sne.editbench import applies  # noqa: E402
from sne.loops import agent_steps  # noqa: E402
def trace_path(run, iid):
    for b in ("artifacts_local/colab", "artifacts_local/kaggle"):
        p = os.path.join(b, run.lower(), "results", "traces", f"trace_{iid}.json")
        if os.path.exists(p): return p
cat = collections.Counter(); detail = collections.Counter()
for h in ("dev", "test"):
    for l in open(f"artifacts_local/FEB/feb_{h}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        t = json.load(open(trace_path(r["run"], r["iid"]), encoding="utf-8"))
        a = next(c.get("arguments") or {} for i, s in agent_steps(t) for c in (s.get("tool_calls") or []) if c.get("function_name") == "edit_file")
        old, new, content = a.get("old_string") or "", a.get("new_string") or "", r["content"]
        out = applies(content, old, new)
        if out.startswith("ok"):
            cat["applied"] += 1; continue
        if out != "not_found":
            cat[out] += 1; continue
        if '\\"' in old and '\\"' not in content:
            cat["not_found: escape"] += 1; continue
        lines = [x for x in old.split("\n") if x.strip()]
        missing = [x for x in lines if x.strip() not in content]
        if not missing:
            cat["not_found: lines present, whitespace or order differs"] += 1; continue
        fl = [x.strip() for x in content.split("\n") if x.strip()]
        best = max((difflib.SequenceMatcher(None, missing[0].strip(), f).ratio() for f in fl), default=0)
        k = "near copy (>=0.9)" if best >= 0.9 else "altered (0.6-0.9)" if best >= 0.6 else "invented (<0.6)"
        cat[f"not_found: {len(missing)}/{len(lines)} lines missing, first {k}"] += 0
        cat[f"not_found: line {k}"] += 1
tot = sum(cat.values())
for k, v in cat.most_common():
    if v: print(f"{v:4d} {100 * v / tot:5.1f}%  {k}")
print("total", tot)

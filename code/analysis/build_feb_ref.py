"""Reference change per FEB state: the lines the reference fix removes and adds in the state's edited file (stripped, non-blank).
Output (private, git-ignored): artifacts_local/FEB/feb_ref.json {state id: {"minus": [...], "plus": [...]}}; states whose file the
reference fix does not touch get empty lists."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)


def lines(patch, path):
    out, f = dict(minus=[], plus=[]), None
    for l in patch.splitlines():
        if l.startswith("+++ "):
            f = l[4:].strip().removeprefix("b/")
            continue
        if l.startswith("--- ") or l.startswith("@@") or f is None:
            continue
        if f == path and l[:1] in "+-" and l[1:].strip():
            out["minus" if l[0] == "-" else "plus"].append(l[1:].strip())
    return out


ref = {json.loads(l)["instance_id"]: json.loads(l)["patch"] for l in open("data/kaggle/tasks.jsonl", encoding="utf-8")}
res = {}
for h in ("dev", "test"):
    for l in open(f"artifacts_local/FEB/feb_{h}.jsonl", encoding="utf-8"):
        s = json.loads(l)
        fp = (s["filepath"] or "").removeprefix("/workspace/").lstrip("./")
        res[s["id"]] = lines(ref.get(s["iid"], ""), fp)
json.dump(res, open("artifacts_local/FEB/feb_ref.json", "w", encoding="utf-8"))
print(len(res), "states;", sum(1 for v in res.values() if v["plus"] or v["minus"]), "with a reference change in their file")

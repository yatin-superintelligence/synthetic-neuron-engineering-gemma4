"""For EXP-113: each FEB state plus E2B's recorded reasoning for its first edit, ending where the tool name is chosen
('<channel|><|tool_call>call:'), labelled by whether that recorded edit applied (official matcher on the file as it stood)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, "src")
from sne.editbench import applies  # noqa: E402
from sne.loops import agent_steps  # noqa: E402
def tp(run, iid):
    for b in ("artifacts_local/colab", "artifacts_local/kaggle"):
        p = os.path.join(b, run.lower(), "results", "traces", f"trace_{iid}.json")
        if os.path.exists(p): return p
for h in ("dev", "test"):
    out = []
    for l in open(f"artifacts_local/FEB/feb_{h}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        t = json.load(open(tp(r["run"], r["iid"]), encoding="utf-8"))
        step = next(s for i, s in agent_steps(t) if any(c.get("function_name") == "edit_file" for c in (s.get("tool_calls") or [])))
        a = next(c.get("arguments") or {} for c in step["tool_calls"] if c.get("function_name") == "edit_file")
        res = applies(r["content"], a.get("old_string") or "", a.get("new_string") or "")
        out.append(dict(id=r["id"], iid=r["iid"], half=h, decision=r["prompt"] + (step.get("message") or "") + "<channel|><|tool_call>call:",
                        label="applies" if res.startswith("ok") else res))
    with open(f"artifacts_local/FEB/feb_decision_{h}.jsonl", "w", encoding="utf-8") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")
    print(h, len(out), "applies", sum(x["label"] == "applies" for x in out))

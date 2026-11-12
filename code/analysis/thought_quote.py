"""First edit_file call of every scored run: does the thought before it quote code (a fenced block) and contain an old_string line verbatim? Applied against failed, E2B against E4B."""
import json,glob,os,collections
from _edits import applied
off=collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f,encoding="utf-8")): off[x["run"]][x["iid"]]=bool(x["resolved"])
cfg={}
for f in glob.glob("experiments/colab_configs/*.json"):
    try:
        for e in json.load(open(f,encoding="utf-8")):
            if isinstance(e,dict) and "id" in e: cfg[e["id"].upper()]=e
    except Exception: pass
for model in ("gemma-4-e2b","gemma-4-e4b"):
    runs=[r for r in off if cfg.get(r.upper(),{}).get("model","").startswith(model) and len(off[r])>=56]
    C=collections.Counter(); n=collections.Counter()
    for r in runs:
        d=next((os.path.join(b,r.lower(),"results","traces") for b in ("artifacts_local/colab","artifacts_local/kaggle") if os.path.isdir(os.path.join(b,r.lower(),"results","traces"))),None)
        if not d: continue
        for iid in off[r]:
            p=os.path.join(d,f"trace_{iid}.json")
            if not os.path.exists(p): continue
            t=json.load(open(p,encoding="utf-8"))
            for s in t["steps"]:
                calls=[c for c in (s.get("tool_calls") or []) if c.get("function_name")=="edit_file"]
                if not calls: continue
                a=calls[0].get("arguments") or {}
                old=a.get("old_string") or ""
                o=json.dumps(s.get("observation") or {})
                failed=not applied(o)
                msg=s.get("message") or ""
                lines=[l.strip() for l in old.splitlines() if len(l.strip())>=12]
                k="failed" if failed else "applied"
                n[k]+=1
                C[(k,"thought has code block")]+= "```" in msg
                C[(k,"an old_string line is in the thought")]+= any(l in msg for l in lines)
                C[(k,"thought tokens")]+= (s.get("metrics") or {}).get("completion_tokens",0)
                break
    print("==",model)
    for k in ("applied","failed"):
        print(f"  first edit {k:8s} n={n[k]:4d} | thought has a code block {100*C[(k,'thought has code block')]/max(n[k],1):5.1f}% | an old_string line verbatim in the thought {100*C[(k,'an old_string line is in the thought')]/max(n[k],1):5.1f}% | mean tokens {C[(k,'thought tokens')]/max(n[k],1):6.0f}")

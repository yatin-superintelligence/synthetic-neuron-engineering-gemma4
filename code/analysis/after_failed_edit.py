"""What E2B and E4B do after a failed first edit (the next call; whether a later edit applies), over every scored run of 56 or more
tasks: first on the 7 regularly solved tasks, then on all tasks. An edit applied when the harness answered it with status ok
(tools/analysis/_edits.py). Needs the runs' traces under artifacts_local/colab (tools/fetch_artifacts.py colab/<run>)."""
import collections
import glob
import json
import os

from _edits import applied

off = collections.defaultdict(dict)
for f in glob.glob("artifacts_local/kaggle_verify/*/verify_results.json"):
    for x in json.load(open(f, encoding="utf-8")):
        off[x["run"]][x["iid"]] = bool(x["resolved"])
cfg = {}
for f in glob.glob("experiments/colab_configs/*.json"):
    try:
        for e in json.load(open(f, encoding="utf-8")):
            if isinstance(e, dict) and "id" in e:
                cfg[e["id"].upper()] = e
    except Exception:
        pass
SOLV = set("fastapi_14786 requests_6644 rich_2943 fastapi_14301 rich_3006 fastapi_9425 rich_3454".split())
for model in ("gemma-4-e2b", "gemma-4-e4b"):
    runs = [r for r in off if cfg.get(r.upper(), {}).get("model", "").startswith(model) and len(off[r]) >= 56]
    for label, only in (("the 7 regularly solved tasks", SOLV), ("all tasks", None)):
        nxt, nxt_ok, later = collections.Counter(), collections.Counter(), collections.Counter()
        used = first_applied = 0
        for r in runs:
            d = next((os.path.join(b, r.lower(), "results", "traces") for b in ("artifacts_local/colab", "artifacts_local/kaggle")
                      if os.path.isdir(os.path.join(b, r.lower(), "results", "traces"))), None)
            if not d:
                continue
            used += 1
            for iid, ok in off[r].items():
                if only is not None and iid not in only:
                    continue
                p = os.path.join(d, f"trace_{iid}.json")
                if not os.path.exists(p):
                    continue
                seq = []
                for s in json.load(open(p, encoding="utf-8"))["steps"]:
                    for c in (s.get("tool_calls") or []):
                        seq.append((c.get("function_name"), json.dumps(c.get("arguments"), sort_keys=True), not applied(s.get("observation"))))
                ed = [k for k, x in enumerate(seq) if x[0] == "edit_file"]
                if not ed:
                    continue
                k = ed[0]
                if not seq[k][2]:
                    first_applied += 1
                    continue
                nx = seq[k + 1] if k + 1 < len(seq) else None
                name = "(end)" if nx is None else nx[0]
                if nx and nx[0] == "edit_file":
                    name = "edit_file (identical retry)" if nx[1] == seq[k][1] else ("edit_file (applies)" if not nx[2] else "edit_file (fails again)")
                nxt[name] += 1
                nxt_ok[name] += ok
                kind = ("a later edit applies" if any(x[0] == "edit_file" and not x[2] for x in seq[k + 1:])
                        else ("write_file used" if any(x[0] == "write_file" for x in seq[k + 1:]) else "no later applied edit"))
                later[(kind, ok)] += 1
        n = sum(nxt.values())
        print(f"== {model} | {label} | runs used {used} of {len(runs)} | first edit applied: {first_applied} task runs | first edit failed: {n} task runs")
        for name, v in nxt.most_common():
            print(f"   next call {name:28s} {v:4d}  solved {nxt_ok[name]}")
        for kk in sorted(set(x for x, _ in later)):
            print(f"   {kk:24s} task runs {later[(kk, True)] + later[(kk, False)]:4d} solved {later[(kk, True)]}")

"""First-edit bench (FEB): the agent state right before E2B's first edit_file call in every official E2B run, with the edited
file as it stood at that moment, split by the shared dev/test halves of the 58 tasks. On the tasks E2B can solve, a first edit
that applies leads to a solve in 72% of runs and one that fails in 14% (tools/analysis/first_edit.py), so the share of first edits
that apply on these real states is the target measure for edits meant to raise tasks solved.
Output (private, git-ignored; competition content): artifacts_local/FEB/feb_{dev,test}.jsonl with id, iid, run, prompt (rendered
with the official chat template and tools), filepath, content (the file before the edit), orig (the recorded outcome)."""
import collections
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "src")
from sne.editbench import walk  # noqa: E402
from sne.loops import messages_before, render, split_map  # noqa: E402

PER_TASK, MAX_PROMPT = 8, 24000
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
    except Exception:  # noqa: BLE001
        pass
runs = sorted(r for r in off if cfg.get(r.upper(), {}).get("model", "").startswith("gemma-4-e2b") and len(off[r]) >= 56)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("models/gemma-4-E2B-it-qat-unquantized")
split = split_map()
rows = collections.defaultdict(list)
skipped = collections.Counter()
for r in runs:
    d = next((os.path.join(b, r.lower(), "results", "traces") for b in ("artifacts_local/colab", "artifacts_local/kaggle")
              if os.path.isdir(os.path.join(b, r.lower(), "results", "traces"))), None)
    if d is None:
        continue
    for f in sorted(glob.glob(os.path.join(d, "trace_*.json"))):
        iid = os.path.basename(f)[6:-5]
        t = json.load(open(f, encoding="utf-8"))
        try:
            first = next(iter(walk(t, iid)), None)
        except Exception:  # noqa: BLE001
            skipped["walk error"] += 1
            continue
        if first is None:
            skipped["no edit"] += 1
            continue
        i, s, a, before, res = first
        if before is None:
            skipped["file unknown"] += 1
            continue
        prompt = render(tok, {"messages": messages_before(t, i)})
        if len(tok(prompt, add_special_tokens=False)["input_ids"]) > MAX_PROMPT:
            skipped["prompt too long"] += 1
            continue
        rows[(split.get(iid), iid)].append(dict(id=f"{r}:{iid}", iid=iid, run=r, solved=off[r].get(iid), prompt=prompt, filepath=a.get("filepath"),
                                                content=before, orig="applied" if (res or {}).get("status") != "error" else "failed"))
os.makedirs("artifacts_local/FEB", exist_ok=True)
rng = random.Random(112)
stats = {}
for half in ("dev", "test"):
    out = []
    for (h, iid), rs in sorted(rows.items()):
        if h != half:
            continue
        rng.shuffle(rs)
        out += rs[:PER_TASK]
    rng.shuffle(out)
    with open(f"artifacts_local/FEB/feb_{half}.jsonl", "w", encoding="utf-8") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")
    stats[half] = dict(states=len(out), tasks=len({x["iid"] for x in out}), originally_applied=sum(x["orig"] == "applied" for x in out))
print("runs", len(runs), "| skipped", dict(skipped), "|", stats)

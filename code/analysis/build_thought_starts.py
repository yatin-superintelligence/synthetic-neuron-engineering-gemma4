"""Thought-start states for a quote-trigger neuron (EXP-117): the agent state right before any E2B model call in its official runs,
rendered with the official chat template (the prompt ends with the opened thought, '<|channel>thought' and a newline), labelled by the
tool response just before it (read_file, other tool, or none for the first call; at a new model turn the template stops at the turn
opener and E2B writes the thought opener itself, so it is appended) and the call E2B then made.
Output (private, git-ignored; competition content): artifacts_local/EXP-117/thought_starts_{dev,test}.jsonl, split by the shared
dev/test task halves, at most PER_CLASS states per task and previous-tool class, prompts of at most MAX_PROMPT tokens."""
import collections
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "src")
from sne.loops import messages_before, render, split_map  # noqa: E402

PER_CLASS, MAX_PROMPT = 4, 12000
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
rng = random.Random(117)
cand = collections.defaultdict(list)  # (half, iid, prev class) -> [(run, path, step index, prev tool, next tool)]
for r in runs:
    d = next((os.path.join(b, r.lower(), "results", "traces") for b in ("artifacts_local/colab", "artifacts_local/kaggle")
              if os.path.isdir(os.path.join(b, r.lower(), "results", "traces"))), None)
    if d is None:
        continue
    for f in sorted(glob.glob(os.path.join(d, "trace_*.json"))):
        iid = os.path.basename(f)[6:-5]
        t = json.load(open(f, encoding="utf-8"))
        prev = None
        for i, s in enumerate(t["steps"]):
            if s.get("source") != "agent":
                continue
            calls = [c.get("function_name") for c in (s.get("tool_calls") or [])]
            if calls:
                cls = "read_file" if prev == "read_file" else ("none" if prev is None else "other")
                cand[(split.get(iid), iid, cls)].append((r, f, i, prev, calls[0]))
                prev = calls[0]
stats = {}
os.makedirs("artifacts_local/EXP-117", exist_ok=True)
for half in ("dev", "test"):
    out = []
    for (h, iid, cls), xs in sorted(cand.items()):
        if h != half:
            continue
        rng.shuffle(xs)
        kept = 0
        for r, f, i, prev, nxt in xs:
            if kept >= PER_CLASS:
                break
            t = json.load(open(f, encoding="utf-8"))
            p = render(tok, {"messages": messages_before(t, i)})
            if p.endswith("<|turn>model\n"):  # a new model turn: the template stops at the turn opener; E2B writes the thought opener itself
                p += "<|channel>thought\n"
            if not p.endswith("<|channel>thought\n"):
                continue
            n = len(tok(p, add_special_tokens=False)["input_ids"])
            if n > MAX_PROMPT:
                continue
            out.append(dict(id=f"{r}:{iid}:{i}", iid=iid, run=r, step=i, prev=prev, prev_class=cls, next=nxt, n_tokens=n, prompt=p))
            kept += 1
    rng.shuffle(out)
    with open(f"artifacts_local/EXP-117/thought_starts_{half}.jsonl", "w", encoding="utf-8") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")
    stats[half] = dict(states=len(out), tasks=len({x["iid"] for x in out}), by_prev=dict(collections.Counter(x["prev_class"] for x in out)),
                       next_edit=sum(x["next"] == "edit_file" for x in out))
print("runs", len(runs), "|", stats)

"""Bench for EXP-144, the second 31B string-end neuron (D-186), from the exact requests vLLM logged in five 31B runs: EXP-129a,
EXP-131a and EXP-139b (unedited) and EXP-140a and EXP-140d (the first neuron and its control, whose faults include the displaced
ones). Requests are teacher-forced on the unedited model later, so an edited run's request is used only as a sequence of tokens.

A label [i, cls] means: output token i is the next token of class cls (the state that predicts it sits at len(prompt) + i - 1).
  ERR    token i ends a string argument with a separator-fused token instead of the closing token <|"|> (id 52): backtick-comma
         (8347), quote-comma (827) or a plain comma (236764), followed by an argument name and a colon. The string's content is
         complete at this state, so the closing token is the right next token.
  OKC    token i is <|"|> closing a string in a well-formed call (the model already does the right thing).
  SEP    token i is the comma right after a closing token (raising the closing token here would close twice).
  OPEN   token i is <|"|> opening a string.
  CONT, CONT_H, CONT_BT, TXT_BT, TXT_BTC, OTHER as in tools/analysis/build_delim_bench.py.
Faults whose token merges content with the separator (for example a closing bracket, a backtick and a comma) are left out: raising
the closing token there would drop content.
Usage: python tools/analysis/build_delim_bench_v2.py   (writes artifacts_local/EXP-144/delim_bench_v2.jsonl.gz)"""
import collections
import gzip
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import served_prompts as SP  # noqa: E402

RUNS = ("exp-129a", "exp-131a", "exp-139b", "exp-140a", "exp-140d")
OUT = os.path.join("artifacts_local", "EXP-144", "delim_bench_v2.jsonl.gz")
Q, TC0, TC1 = 52, 48, 49
FAULT = {8347: "backtick-comma", 827: "quote-comma", 236764: "comma"}
TICK = chr(96)
KEY = re.compile(r"^(old_string|new_string|filepath|allow_multiple|content|command):")
HARD_END = ("\n", ",", ")", ":", "]", "}", chr(34), chr(39), "n")
PER_CALL, GOOD_EDITS, OTHER_CALLS = 6, 140, 70
rng = random.Random(144)


def label(ids, txt):
    lab, n = [], len(ids)
    regions, i = [], 0
    while i < n:
        if ids[i] == TC0:
            j = next((k for k in range(i + 1, n) if ids[k] == TC1), n)
            regions.append((i, j))
            i = j
        i += 1
    inside, faults = set(), []
    for a, b in regions:
        inside.update(range(a, min(b, n - 1) + 1))
        err = None
        for k in range(a + 1, b):
            # a string is open at k (an odd number of string tokens since the call began): unquoted numbers and booleans
            # are followed by a plain comma and the next argument legitimately
            open_string = sum(1 for j in range(a + 1, k) if ids[j] == Q) % 2 == 1
            if ids[k] in FAULT and ids[k - 1] != Q and open_string and KEY.match("".join(txt[k + 1:k + 8]).lstrip(",")):
                err = k
                break
        if err is not None:
            lab.append([err, "ERR"])
            faults.append(FAULT[ids[err]])
            continue
        qs = [k for k in range(a + 1, b) if ids[k] == Q]
        if len(qs) % 2 or not qs:
            continue
        for m in range(0, len(qs), 2):
            o, c = qs[m], qs[m + 1]
            lab.append([o, "OPEN"])
            lab.append([c, "OKC"])
            if c + 1 < b and txt[c + 1].startswith(","):
                lab.append([c + 1, "SEP"])
            cont = list(range(o + 1, c))
            hard = [k for k in cont if txt[k - 1].endswith(HARD_END)]
            for k in cont:
                if TICK in txt[k]:
                    lab.append([k, "CONT_BT"])
            for k in rng.sample(hard, min(12, len(hard))):
                lab.append([k, "CONT_H"])
            for k in rng.sample(cont, min(8, len(cont))):
                lab.append([k, "CONT"])
    outside = [k for k in range(1, n) if k not in inside]
    for k in outside:
        if TICK in txt[k]:
            lab.append([k, "TXT_BTC" if ids[k] == 8347 else "TXT_BT"])
    for k in rng.sample(outside, min(6, len(outside))):
        lab.append([k, "OTHER"])
    seen, out = set(), []
    for k, c in lab:
        if k not in seen:
            seen.add(k)
            out.append([k, c])
    return sorted(out), faults


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join("models", "gemma-4-E2B-it-qat-unquantized"))
    assert tok.convert_tokens_to_ids("<|" + chr(34) + "|>") == Q and tok.decode([8347]) == TICK + ","
    cache = {}
    rows, by_call = [], collections.defaultdict(list)
    for run in RUNS:
        matched, stat, _ = SP.match(os.path.join("artifacts_local", "colab", run))
        print(run, "steps matched", stat.get("matched to exactly one request"), "of", stat.get("agent steps"))
        for r in matched:
            ids = r.get("output_ids") or []
            if TC0 not in ids or not r.get("prompt_ids"):
                continue
            txt = [cache.setdefault(t, tok.decode([t])) for t in ids]
            lab, faults = label(ids, txt)
            if not lab:
                continue
            row = dict(run=run, iid=r["iid"], step=r["step"], calls=r["calls"], prompt_ids=r["prompt_ids"], output_ids=ids, labels=lab,
                       fault=faults[0] if faults else None)
            rows.append(row)
            if row["fault"]:
                s0 = ids.index(TC0)
                by_call[(run, r["iid"], tuple(ids[s0:s0 + 40]))].append(row)
    keep = []
    for k, rs in sorted(by_call.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        step = max(1, len(rs) // PER_CALL)
        keep += rs[::step][:PER_CALL]
    good = [r for r in rows if not r["fault"]]
    edits = [r for r in good if "edit_file" in r["calls"]]
    others = [r for r in good if "edit_file" not in r["calls"]]
    rng.shuffle(edits)
    rng.shuffle(others)
    writes = [r for r in others if "write_file" in r["calls"]][:25]
    rest = [r for r in others if "write_file" not in r["calls"]][:OTHER_CALLS - len(writes)]
    keep += edits[:GOOD_EDITS] + writes + rest
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    cls = collections.Counter(c for r in keep for _, c in r["labels"])
    print(f"\nrequests with a fault: {sum(bool(r['fault']) for r in rows)} ({dict(collections.Counter(r['fault'] for r in rows if r['fault']))}); distinct faulty calls {len(by_call)}")
    print(f"kept {len(keep)} requests: faulty {sum(bool(r['fault']) for r in keep)} ({dict(collections.Counter(r['fault'] for r in keep if r['fault']))}), "
          f"well-formed edit_file {len(edits[:GOOD_EDITS])}, write_file {len(writes)}, other {len(rest)}")
    print("labelled positions by class:", dict(sorted(cls.items())))
    per = collections.defaultdict(collections.Counter)
    for r in keep:
        if r["fault"]:
            per[r["iid"]][r["run"] + " " + r["fault"]] += 1
    for t, c in sorted(per.items()):
        print(f"  {t:16s} {dict(c)}")
    print("tokens:", sum(len(r["prompt_ids"]) + len(r["output_ids"]) for r in keep), "| wrote", OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()

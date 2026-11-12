"""Builds the bench for EXP-140, 31B's string-delimiter fault, from the exact requests vLLM logged in two unedited 31B runs
(EXP-129a and EXP-131a: prompt token ids as served, output token ids as generated).

In Gemma 4's tool-call format every string argument is opened and closed by one token, <|"|> (id 52):
    call:edit_file{filepath:<|"|>a.py<|"|>,new_string:<|"|>...<|"|>,old_string:<|"|>...<|"|>}
31B sometimes ends a string with the single token backtick-comma (id 8347: a backtick and the comma that should follow the closing
token) and goes on with the next argument's name. The parser then never sees the string end, the call loses an argument and is
refused.

Every row is one served request: prompt_ids, output_ids and labelled positions. A label [i, cls] means: output token i is the next
token of class cls, so the state to read is the one at absolute index len(prompt_ids) + i - 1. Classes:
  ERR      output token i is backtick-comma used as a string end (the next tokens spell an argument name and a colon)
  ERR2     the same with a longer token that ends in backtick-comma (for example a closing bracket, a backtick and a comma):
           content and string end merged into one token
  OKC      token i is <|"|> closing a string in a well-formed call
  OPEN     token i is <|"|> opening a string in a well-formed call
  CONT     token i is content inside a string of a well-formed call (a sample); CONT_H when the token before it ends the way the
           tokens before the faults end (a newline, a comma, a bracket, a colon, a quote); CONT_BT when token i holds a backtick
  TXT_BT   token i holds a backtick outside any tool call (markdown in the thought or the text); TXT_BTC when it is exactly
           backtick-comma
  OTHER    a sample of all other positions outside tool calls
Usage: python tools/analysis/build_delim_bench.py   (writes artifacts_local/EXP-140/delim_bench.jsonl.gz and prints the counts)"""
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

RUNS = ("exp-129a", "exp-131a")
OUT = os.path.join("artifacts_local", "EXP-140", "delim_bench.jsonl.gz")
Q, TC0, TC1, BTC = 52, 48, 49, 8347
TICK = chr(96)
KEY = re.compile(r"^[a-z_]{3,20}:")
HARD_END = ("\n", ",", ")", ":", "]", "}", chr(34), chr(39), "n")
PER_CALL, PER_CALL2, GOOD_EDITS, OTHER_CALLS = 6, 4, 140, 70
rng = random.Random(140)


def label(ids, txt):
    """([[i, cls], ...], text of the first faulty call or None) for one output."""
    lab, n = [], len(ids)
    regions, i = [], 0
    while i < n:
        if ids[i] == TC0:
            j = next((k for k in range(i + 1, n) if ids[k] == TC1), n)
            regions.append((i, j))
            i = j
        i += 1
    inside = set()
    faulty = None
    for a, b in regions:
        inside.update(range(a, min(b, n - 1) + 1))
        err = None
        for k in range(a + 1, b):
            if txt[k].endswith(TICK + ",") and KEY.match("".join(txt[k + 1:k + 8])):
                err = k
                break
        if err is not None:
            lab.append([err, "ERR" if ids[err] == BTC else "ERR2"])
            faulty = faulty or "".join(txt[a:b])
            continue
        qs = [k for k in range(a + 1, b) if ids[k] == Q]
        if len(qs) % 2 or not qs:
            continue  # not well formed in another way, or no string argument: no labels
        for m in range(0, len(qs), 2):
            o, c = qs[m], qs[m + 1]
            lab.append([o, "OPEN"])
            lab.append([c, "OKC"])
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
            lab.append([k, "TXT_BTC" if ids[k] == BTC else "TXT_BT"])
    for k in rng.sample(outside, min(6, len(outside))):
        lab.append([k, "OTHER"])
    seen, out = set(), []
    for k, c in lab:  # one class per position; the first label wins (ERR, OPEN, OKC, CONT_BT, CONT_H, CONT, TXT_*, OTHER)
        if k not in seen:
            seen.add(k)
            out.append([k, c])
    return sorted(out), faulty


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join("models", "gemma-4-E2B-it-qat-unquantized"))  # the Gemma 4 sizes share one vocabulary
    assert tok.convert_tokens_to_ids("<|" + chr(34) + "|>") == Q and tok.decode([BTC]) == TICK + ","
    cache = {}
    rows, by_call = [], collections.defaultdict(list)
    for run in RUNS:
        matched, stat, _ = SP.match(os.path.join("artifacts_local", "colab", run))
        print(run, dict(stat))
        for r in matched:
            ids = r.get("output_ids") or []
            if TC0 not in ids or not r.get("prompt_ids"):
                continue
            txt = [cache.setdefault(t, tok.decode([t])) for t in ids]
            lab, faulty = label(ids, txt)
            if not lab:
                continue
            kinds = collections.Counter(c for _, c in lab)
            row = dict(run=run, iid=r["iid"], step=r["step"], calls=r["calls"], prompt_ids=r["prompt_ids"], output_ids=ids, labels=lab,
                       fault=("ERR" if kinds["ERR"] else "ERR2" if kinds["ERR2"] else None), call=faulty)
            rows.append(row)
            if row["fault"]:
                by_call[(run, r["iid"], faulty)].append(row)
    keep = []
    for (run, iid, call), rs in sorted(by_call.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
        cap = PER_CALL if rs[0]["fault"] == "ERR" else PER_CALL2
        step = max(1, len(rs) // cap)
        keep += rs[::step][:cap]
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
            f.write(json.dumps({**{k: v for k, v in r.items() if k != "call"}, "call_chars": len(r["call"] or "")}) + "\n")
    cls = collections.Counter(c for r in keep for _, c in r["labels"])
    print(f"\nrequests with a tool call and labels: {len(rows)} | with a fault: {sum(bool(r['fault']) for r in rows)} "
          f"({sum(r['fault'] == 'ERR' for r in rows)} ERR, {sum(r['fault'] == 'ERR2' for r in rows)} ERR2) | distinct faulty calls {len(by_call)}")
    print(f"kept {len(keep)} requests: faulty {sum(bool(r['fault']) for r in keep)}, well-formed edit_file {len(edits[:GOOD_EDITS])}, "
          f"write_file {len(writes)}, other tools {len(rest)}")
    print("labelled positions by class:", dict(sorted(cls.items())))
    ptok = sorted(len(r["prompt_ids"]) + len(r["output_ids"]) for r in keep)
    print(f"tokens per request: median {ptok[len(ptok) // 2]}, max {ptok[-1]}, total {sum(ptok)}")
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["fault"]:
            per[r["iid"]][r["run"] + " " + r["fault"]] += 1
    kept = collections.Counter(r["iid"] for r in keep if r["fault"] == "ERR")
    print("faulty requests by task (all found; ERR kept):")
    for t, c in sorted(per.items()):
        print(f"  {t:16s} {dict(c)} | ERR kept {kept[t]}")
    print("wrote", OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()

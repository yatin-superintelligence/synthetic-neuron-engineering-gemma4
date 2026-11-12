"""EXP-079: what would Gemma 4 31B do at E2B's own agent states? Teacher labels for synthetic decision neurons.

E2B's official runs record every agent state: the conversation so far and the action E2B took next (a tool name, or stopping with a
text answer). This script asks the official 31B model (W4A16, vLLM, the official stack) to continue each of those states with its own
thinking and action, 4 samples each. Where 31B reliably does something else than E2B, that decision is a candidate for a synthetic
neuron in E2B: a read-out that detects the state and a write that moves E2B towards 31B's action (as in EXP-076, closed form, no
gradient training). Nothing is fitted here; this only produces the labels and their counts.

Fixed before the run:
- States: every agent step of the three unedited E2B official runs (exp029, exp037, exp-044a) that made a tool call, plus every step
  where the main agent (swe_baseline_agent) answered with text and no tool call ("STOP").
- Prompt: the trace's own conversation up to the state (sne.loops.messages_before), rendered with the same chat template and tool
  declarations as the replay benchmarks. The agent ran with event compaction, so long histories are cut to the system and task
  messages plus the most recent messages that fit 22,000 tokens, starting at an assistant turn.
- Teacher sampling: 4 samples, temperature 1.0, top_p 0.95, top_k 64 (Gemma's defaults), at most 3,072 new tokens, stopping at the
  end of the first tool call. The action is the first tool name after <|tool_call>call:, STOP when the model ends its turn without a
  tool call, and UNDECIDED when it runs out of tokens first.
- Output: artifacts_local/EXP-079/teacher_labels.jsonl (task text stays out of git; rows hold ids, actions and token counts only).
Usage on a runtime with the official stack and the 31B model: /opt/sne/bin/python exp079_teacher_decisions.py [--smoke]
"""
import collections
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from sne import loops as L  # noqa: E402  (standard library only)

SMOKE = "--smoke" in sys.argv
MODEL = os.environ.get("SNE_TEACHER", "/content/kaggle_input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2")
def _traces(r):  # a run made on Kaggle, or a Colab run (EXP-083 labels E4B's Colab runs)
    k = os.path.join(REPO, "artifacts_local", "kaggle", r, "results", "traces")
    return k if os.path.isdir(k) else os.path.join(REPO, "artifacts_local", "colab", r, "results", "traces")


# defaults are EXP-079's (the three unedited E2B runs); EXP-083 sets these two for E4B
RUNS = {r: _traces(r) for r in os.environ.get("SNE_TEACHER_RUNS", "exp029,exp037,exp-044a").split(",")}
OUT = os.path.join(REPO, "artifacts_local", os.environ.get("SNE_TEACHER_OUT", "EXP-079"))
MAX_PROMPT, MAX_NEW, N = 22000, 3072, 4
CALL = re.compile(r"<\|tool_call>call:([A-Za-z_]+)")


def states():
    split = L.split_map()
    rows = []
    for run, d in RUNS.items():
        for f in sorted(glob.glob(os.path.join(d, "trace_*.json"))):
            iid = os.path.basename(f)[6:-5]
            t = json.load(open(f, encoding="utf-8"))
            for i, s in L.agent_steps(t):
                calls = s.get("tool_calls") or []
                ex = s.get("extra") or {}
                if calls:
                    act = calls[0].get("function_name")
                elif ex.get("author") == "swe_baseline_agent" and ex.get("event_type") == "final":
                    act = "STOP"
                else:
                    continue
                rows.append(dict(run=run, iid=iid, step=i, split=split.get(iid, "dev"), author=ex.get("author"), e2b=act, trace=f))
    return rows


def prompt(tok, t, step):
    msgs = L.messages_before(t, step)
    head, tail = msgs[:2], msgs[2:]
    render = lambda m: L.render(tok, dict(messages=m))  # noqa: E731
    text = render(head + tail)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    cut = 0
    while len(ids) > MAX_PROMPT and cut < len(tail):
        cut += 1
        while cut < len(tail) and tail[cut]["role"] != "assistant":
            cut += 1
        text = render(head + tail[cut:])
        ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids, cut


def action(text, finish):
    m = CALL.search(text)
    if m:
        return m.group(1)
    return "UNDECIDED" if finish == "length" else "STOP"


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    rows = states()
    if SMOKE:
        rows = rows[:6] + [r for r in rows if r["e2b"] == "STOP"][:2]
    print("[teacher] states", len(rows), collections.Counter(r["e2b"] for r in rows).most_common(), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    cache, prompts = {}, []
    for r in rows:
        t = cache.setdefault(r["trace"], json.load(open(r["trace"], encoding="utf-8")))
        ids, cut = prompt(tok, t, r["step"])
        r.update(prompt_tokens=len(ids), cut_messages=cut)
        prompts.append(ids)
    print("[teacher] prompt tokens: median", sorted(r["prompt_tokens"] for r in rows)[len(rows) // 2], "max", max(r["prompt_tokens"] for r in rows), flush=True)
    llm = LLM(model=MODEL, max_model_len=MAX_PROMPT + MAX_NEW + 512, gpu_memory_utilization=0.92, enable_prefix_caching=True,
              max_num_seqs=64, dtype="bfloat16")
    sp = SamplingParams(n=N, temperature=1.0, top_p=0.95, top_k=64, max_tokens=MAX_NEW, stop=["<tool_call|>"],
                        include_stop_str_in_output=True, skip_special_tokens=False, seed=79)
    # in chunks, each written as soon as it is done, so a failure keeps what was finished; the first chunk is small, so a broken
    # pipeline shows within minutes. Rows are in trace order, so a chunk holds whole stretches of tasks and prefix caching still helps.
    t0 = time.time()
    bounds = [0, min(8, len(rows))] + list(range(8 + 250, len(rows), 250)) + [len(rows)]
    bounds = sorted(set(bounds))
    with open(os.path.join(OUT, "teacher_labels.jsonl"), "w", encoding="utf-8") as fo:
        for a, b in zip(bounds, bounds[1:]):
            outs = llm.generate([dict(prompt_token_ids=p) for p in prompts[a:b]], sp, use_tqdm=False)
            for r, o in zip(rows[a:b], outs):
                acts = [action(c.text, c.finish_reason) for c in o.outputs]
                r.update(teacher=acts, teacher_tokens=[len(c.token_ids) for c in o.outputs])
                fo.write(json.dumps({k: v for k, v in r.items() if k != "trace"}) + "\n")
            fo.flush()
            done = rows[:b]
            print(f"[teacher] {b}/{len(rows)} states in {time.time() - t0:.0f} s | majority agrees with E2B in",
                  sum(collections.Counter(r["teacher"]).most_common(1)[0][0] == r["e2b"] for r in done), f"of {len(done)}", flush=True)
            if a == 0:
                for r in rows[:b]:
                    print("[teacher] check", r["e2b"], "->", r["teacher"], r["teacher_tokens"], flush=True)
    agree = collections.Counter()
    pairs = collections.Counter()
    for r in rows:
        maj, k = collections.Counter(r["teacher"]).most_common(1)[0]
        agree[maj == r["e2b"]] += 1
        if k >= 3 and maj != r["e2b"]:
            pairs[(r["e2b"], maj)] += 1
    summary = dict(states=len(rows), teacher_majority_agrees_with_e2b=agree[True], disagrees=agree[False],
                   confident_disagreements_e2b_to_teacher={f"{a} -> {b}": n for (a, b), n in pairs.most_common(20)},
                   teacher_actions=collections.Counter(a for r in rows for a in r["teacher"]).most_common(),
                   e2b_actions=collections.Counter(r["e2b"] for r in rows).most_common())
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1)
    print("[teacher] SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

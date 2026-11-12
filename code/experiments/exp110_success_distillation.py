"""EXP-110: E2B fine-tuned on officially verified successful agent runs of its own and larger models, evaluated split-half.

Why: training-free edits change behaviour but not tasks solved (D-104 to D-113); the author asked what could give a 50% gain. The
earlier SFT (EXP-088, D-079) used scripted one-path trajectories and lowered tasks solved. Here the data are real agent runs that the
official verification scored as resolved: E4B (unedited and with teacher neurons), 12B, 31B and E2B itself, on the 58 tasks.
Contamination is avoided by halves: the 58 tasks' shared dev/test split (sne.loops.split_map); a model trained on one half's
successes is evaluated only on the other half's tasks (EXP-110a: trained on dev, evaluated on test; EXP-110b: trained on test,
evaluated on dev), so together they evaluate all 58 tasks with none seen in training.
Data per half (scratch/analysis/build_success_sft.py): every resolved run's full conversation in chat-template form with reasoning
kept; at most 8 runs per task (seed 110), so a few easy tasks do not dominate; tool results cut to 3,000 characters each so whole
runs fit in 12,288 tokens. Training: EXP-088's recipe (LoRA rank 16, alpha 32 on gate, up, down and o_proj of every layer; 2 epochs;
AdamW 1e-4; loss on assistant turns only), then the vLLM load check on the official W4A16 checkpoint.
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HALF = os.environ.get("HALF", "dev")  # train on this half's successes
EXPID = "EXP-110a" if HALF == "dev" else "EXP-110b"
_src = open(os.path.join(HERE, "exp088_sft_lora.py"), encoding="utf-8").read()
E88 = {"__file__": os.path.join(HERE, "exp088_sft_lora.py"), "__name__": "exp088lib"}
exec(compile(_src.split('with Experiment("EXP-088"')[0], "exp088", "exec"), E88)
E88["OUT"] = os.path.join(ROOT, "artifacts_local", EXPID)
E88["MAXLEN"] = int(os.environ.get("MAXLEN", 16384))  # successful runs are long (median about 12,600 tokens on dev); tool results are also cut to CUT characters
CAP, CUT = 8, 1500


def prepare(messages):
    """Tool results cut to CUT characters; consecutive assistant messages (a turn without a tool call followed by another turn) merged
    into one turn, as the chat template renders them, so every assistant turn's tokens can be located for the loss."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "tool" and len(m.get("content") or "") > CUT:
            m["content"] = m["content"][:CUT] + " [cut]"
        if m.get("role") == "assistant" and not m.get("tool_calls") and m.get("content"):
            # text without a tool call is kept as reasoning: the generation prompt always opens the thought channel, and a
            # content-only turn renders without it, so its tokens could not be located
            m = {"role": "assistant", "content": "", "reasoning": m["content"]}
        if m.get("role") == "assistant" and out and out[-1].get("role") == "assistant":
            prev = out[-1]
            text = (chr(10) * 2).join(x for x in (prev.get("reasoning") or prev.get("content") or "", m.get("reasoning") or m.get("content") or "") if x)
            merged = {"role": "assistant", "content": "" if m.get("tool_calls") else text}
            if m.get("tool_calls"):
                merged.update(reasoning=text, tool_calls=m["tool_calls"])
            out[-1] = merged
            continue
        out.append(m)
    # an assistant turn without a tool call (final summary, narration not followed by another turn) renders its text outside the
    # thought channel while the generation prompt opens one, so its tokens cannot be located: such turns are dropped
    return [m for m in out if m.get("role") != "assistant" or m.get("tool_calls")]

with Experiment(EXPID, f"success-distillation-train-{HALF}", category="training", depends_on=["EXP-088", "EXP-103"],
                motivation="Fine-tune E2B on real, officially resolved agent runs (its own and larger models') from one half of the 58 tasks; "
                           "evaluate on the other half only.",
                hypothesis="Trained on the other half's successful runs, E2B solves more tasks of this half than unedited E2B at the same seed.",
                config=dict(half=HALF, cap_per_task=CAP, tool_chars=CUT, rank=E88["R"], alpha=E88["ALPHA"], lr=E88["LR"], epochs=E88["EPOCHS"],
                            max_len=E88["MAXLEN"])) as ex:
    os.makedirs(E88["OUT"], exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "artifacts_local", "EXP-110", f"success_{HALF}.jsonl"), encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["iid"]].append(r)
    rng = random.Random(110)
    pick = []
    for iid, rs in sorted(by.items()):
        rng.shuffle(rs)
        pick += rs[:CAP]
    if SMOKE:
        pick = pick[:8]
    examples = []
    for r in pick:
        examples.append(dict(instance_id=f"{r['iid']}__{r['run']}", messages=prepare(r["messages"])))
    ex.log("runs", len(rows), "| tasks", len(by), "| used", len(examples), "| by model", dict(collections.Counter(r["model"] for r in pick)))
    hist, dst, n_tr, n_val = E88["train"](examples, ex)
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    check = E88["vllm_check"](dst, ex)
    ex.result(half=HALF, tasks=len(by), trajectories=len(examples), train=n_tr, val=n_val, history=hist, vllm_check=check, adapter=dst,
              headline=f"SFT on {n_tr} resolved runs of the {HALF} half: validation loss {hist['val_before']} -> {hist['val'][-1][1] if hist['val'] else None}")

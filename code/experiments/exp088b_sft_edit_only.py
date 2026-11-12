"""EXP-088b (training track, NOT training-free): SFT on the fix step only.

EXP-088a (the EXP-088 adapter, official stack, seed 1) resolved 0 of 56 tasks on Kaggle. Its traces show why: trained on whole
scripted trajectories (run the failing tests, grep, read, edit, rerun, submit), the agent ran run_command in 63% of its tool calls
(unedited 9% to 13%), made 2,782 model calls (unedited about 650) and failed 84% of its edits, many on files it never read. It did
learn one thing the unedited model lacks: its old_strings carried backslash-quote 5 times against about 16 a run unedited.

This run keeps what helped and drops what hurt. Same tasks, same trajectories as context, same LoRA (rank 16, alpha 32, MLP and o_proj
of every layer), 2 epochs, lr 1e-4; but the loss is on the edit_file turn only (its reasoning and its tool call), and each example
ends at that turn. The exploring policy (when to run, read, search, submit) gets no gradient. Pre-registered (D-072): the adapter
must load in the official vLLM; then one official agent run at seed 1 on the Colab L4 (EXP-088c), scored on Kaggle, against the
unedited E2B at seed 1 on the same hardware (EXP-044g).
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

_src = open(os.path.join(os.path.dirname(__file__), "exp088_sft_lora.py"), encoding="utf-8").read()
G = {"__file__": os.path.join(os.path.dirname(__file__), "exp088_sft_lora.py"), "__name__": "exp088"}
exec(compile(_src.split('with Experiment("EXP-088"')[0], "exp088", "exec"), G)
G["OUT"] = os.path.join(ROOT, "artifacts_local", "EXP-088b")


def encode_edit(tok, msgs):
    """Token ids and labels for the trajectory up to its edit_file turn; loss on that turn only."""
    k = next((i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")
              and m["tool_calls"][0]["function"]["name"] == "edit_file"), None)
    if k is None:
        return None, None
    msgs = msgs[:k + 1]
    full = G["render"](tok, msgs, False)
    a_txt = G["render"](tok, msgs[:k], True)
    if not full.startswith(a_txt):
        return None, None
    ids = tok(full, add_special_tokens=False)["input_ids"]
    a = len(tok(a_txt, add_special_tokens=False)["input_ids"])
    labels = [-100] * a + ids[a:]
    return ids, labels


G["encode"] = encode_edit

with Experiment("EXP-088b", "sft-edit-only", category="training", depends_on=["EXP-088", "EXP-088a"],
                motivation="EXP-088's whole-trajectory SFT taught a bad exploring policy (0 of 56 solved) but unescaped old_strings; training only the "
                           "edit turn keeps the fix-writing skill without changing how the agent explores.",
                hypothesis="An SFT LoRA trained only on the edit_file turn raises E2B's official tasks solved at seed 1 against unedited E2B at seed 1 "
                           "on the same hardware, with fewer failed edits.",
                config=dict(rank=G["R"], alpha=G["ALPHA"], lr=G["LR"], epochs=G["EPOCHS"], accumulation=G["ACC"], warmup=G["WARM"], max_len=G["MAXLEN"],
                            target=G["TARGET"], base=G["QAT"], loss="edit_file turn only", context="trajectory up to and including the edit turn")) as ex:
    os.makedirs(G["OUT"], exist_ok=True)
    os.makedirs(G["WORK"], exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "artifacts_local", "trainset", "*", "trainset", "tasks.jsonl")))
    tasks = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
    tasks = [t for t in tasks if t.get("source") == "synthetic" and isinstance(t.get("buggy_excerpt"), dict) and "content" in t["buggy_excerpt"]]
    uniq = {}
    for t in tasks:
        key = hashlib.sha1(t["bug_patch"].encode("utf-8")).hexdigest()[:8]
        if key not in uniq:
            t["instance_id"] = f"{t['instance_id']}_{key}"
            uniq[key] = t
    tasks = list(uniq.values())
    ex.log("training tasks", len(tasks), "| per repo", {r: sum(t["repo"] == r for t in tasks) for r in sorted({t["repo"] for t in tasks})})
    if SMOKE:
        tasks = tasks[:12]
    examples, skipped = G["build_data"](tasks, ex)
    hist, dst, n_tr, n_val = G["train"](examples, ex)
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    check = G["vllm_check"](dst, ex)
    ex.result(tasks=len(tasks), trajectories=len(examples), skipped=skipped, train=n_tr, val=n_val, history=hist, vllm_check=check, adapter=dst,
              headline=f"SFT (edit turn only) LoRA r{G['R']} on {n_tr} examples: validation loss {hist['val_before']} -> "
                       f"{hist['val'][-1][1] if hist['val'] else None}")

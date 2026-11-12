"""EXP-086 build check: is the deeper checkpoint served by vLLM the model we meant?

Independent of the checkpoint builder and of vLLM: loads the public bf16 E2B (QAT unquantized) in transformers on the CPU and runs
the decoder layers in an explicit order (a block run twice, each copy with its own per-layer embedding slice, layers 13 and 14 still
supplying the shared keys and values), then decodes greedily. The first tokens are compared with the answers vLLM gave for the same
MBPP+ dev problems in the EXP-086 scan (W4A16, so small numerical differences are expected; a collapse in one and not the other
would mean the build or vLLM's reading of it is wrong).

Usage (on the EXP-086 Colab VM, official Python): python experiments/scripts/exp086_build_check.py <tag> [<tag> ...]
  tag: identity, or dup_s<S>_k<K> (for example dup_s3_k2). Output: artifacts_local/EXP-086/build_check.json
"""
import json
import os
import sys
from collections import UserDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402
from sne import ROOT, bench  # noqa: E402
from sne.evaluate import render  # noqa: E402

OUT = os.path.join(ROOT, "artifacts_local", "EXP-086")
N_TASKS, N_NEW = 3, 24
torch.set_grad_enabled(False)
torch.set_num_threads(10)


def order_for(tag):
    if tag == "identity":
        return list(range(35))
    s, k = int(tag.split("_s")[1].split("_")[0]), int(tag.split("_k")[1])
    return list(range(s + k)) + list(range(s, s + k)) + list(range(s + k, 35))


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    src = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/bf16_src")
    tok = AutoTokenizer.from_pretrained(src)
    m = Gemma4ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16).eval()
    lm = m.model.language_model
    cfg = lm.config

    def next_logits(ids, order):
        emb = lm.embed_tokens(ids)
        ple = lm.project_per_layer_inputs(emb, lm.get_per_layer_inputs(ids, emb))
        pos = torch.arange(ids.shape[1])[None]
        kw = dict(config=cfg, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos)
        masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        pe = {t: lm.rotary_emb(emb, pos, t) for t in set(cfg.layer_types)}
        shared, h = UserDict(), emb
        for j in order:
            t = cfg.layer_types[j]
            h = lm.layers[j](h, ple[:, :, j, :], shared_kv_states=shared, position_embeddings=pe[t], attention_mask=masks[t],
                             position_ids=pos, past_key_values=None)
        h = lm.norm(h[:, -1:])
        x = m.lm_head(h)[0, -1].float()
        cap = cfg.final_logit_softcapping
        return torch.tanh(x / cap) * cap if cap else x

    dev = bench.mbppplus("dev")[:N_TASKS]
    prompts = render(tok, [t["user"] for t in dev])
    out = {}
    for tag in sys.argv[1:]:
        order = order_for(tag)
        saved = os.path.join(OUT, f"rows_{tag}_dev.json")
        vllm = {r["task_id"]: r["response"] for r in json.load(open(saved))} if os.path.exists(saved) else {}
        out[tag] = []
        for t, p in zip(dev, prompts):
            ids = tok(p, return_tensors="pt", add_special_tokens=True)["input_ids"]
            new = []
            for _ in range(N_NEW):
                nxt = int(next_logits(ids, order).argmax())
                new.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
            txt = tok.decode(new)
            out[tag].append(dict(task_id=t["task_id"], transformers_bf16=txt, vllm_w4a16=(vllm.get(t["task_id"]) or "")[:200]))
            print(f"[check] {tag} {t['task_id']}\n  transformers: {txt!r}\n  vLLM:         {(vllm.get(t['task_id']) or '')[:120]!r}", flush=True)
        json.dump(out, open(os.path.join(OUT, "build_check.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

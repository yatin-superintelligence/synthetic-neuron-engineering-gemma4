"""EXP-003: does llama.cpp's Gemma 4 implementation compute the same function as the Hugging Face reference?

Gemma 4 is a new architecture (per-layer embeddings, KV sharing, double-wide MLPs, sandwich norms). A silent difference
in the fast engine would invalidate transferring edits derived in PyTorch.

Method (teacher-forced, replacing the padded-batch comparison that failed on the Level-Zero driver, attempts 1-2 in
_failed): for 40 prompts (MBPP original train split, never scored), take llama.cpp's greedy continuation of 128 tokens
at F16 and Q4_0. Feed prompt + continuation through the HF bf16 model at batch size 1 and record, at every generated
position, whether HF's argmax equals llama.cpp's token, and the HF probability of that token.
Reference arm: the same measure with bf16 HF against itself would be 100% by construction, so the meaningful baseline is
Q4_0 vs F16 disagreement, and a random-token control shows the metric is not trivially high.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch  # noqa: E402

from sne import DATA, MODELS  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.evaluate import render  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

N, NEW = 40, 128
HF = os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized")

with Experiment("EXP-003", "engine-fidelity-hf-vs-llamacpp", category="infrastructure", depends_on=["EXP-001"],
                motivation="The fast engine must compute the same function as the reference implementation, otherwise "
                           "edits derived in PyTorch would not transfer.",
                hypothesis="HF agrees with llama.cpp F16 on over 98% of teacher-forced next tokens, and slightly less "
                           "with Q4_0 continuations.",
                config=dict(n_prompts=N, new_tokens=NEW, source="mbpp_full_train first 40 by file order",
                            method="teacher-forced argmax agreement, HF bf16 batch size 1")) as ex:
    rows = [json.loads(l) for l in open(os.path.join(DATA, "mbpp_full_train.jsonl"), encoding="utf-8")][:N]
    users = [f"{r['text']}\nYour code should pass this test:\n{r['test_list'][0]}" for r in rows]
    tok = tokenizer()
    prompts = render(tok, users)
    gens = {}
    for prec in ["f16", "q4_0"]:
        with LlamaServer(GGUF[prec], parallel=8, log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            g, st = srv.complete_many(prompts, max_tokens=NEW)
        gens[prec] = [x["text"] for x in g]
        ex.save_json(f"generations_{prec}.json", gens[prec])
        ex.log(prec, st)
    from sne.model import load
    model, tok2 = load(HF)
    out = {}
    torch.manual_seed(3)
    for prec, texts in gens.items():
        agree = total = 0
        probs, first_dis = [], []
        ctl = 0
        t0 = time.time()
        for p, txt in zip(prompts, texts):
            a = tok("<bos>" + p, add_special_tokens=False)["input_ids"]
            b = tok(txt, add_special_tokens=False)["input_ids"]
            if not b:
                continue
            ids = torch.tensor([a + b], device="xpu")
            with torch.no_grad():
                lg = model(input_ids=ids, use_cache=False).logits[0].float().cpu()
            pred = lg[len(a) - 1: len(a) + len(b) - 1]
            tgt = torch.tensor(b)
            am = pred.argmax(-1)
            agree += int((am == tgt).sum())
            total += len(b)
            pr = pred.softmax(-1)[torch.arange(len(b)), tgt]
            probs.append(float(pr.mean()))
            d = (am != tgt).nonzero()
            first_dis.append(int(d[0]) if len(d) else len(b))
            ctl += int((am == tgt[torch.randperm(len(b))]).sum())
        out[prec] = dict(tokens=total, argmax_agreement=round(agree / total, 5), mean_prob_of_token=round(sum(probs) / len(probs), 4),
                         sequences_fully_agreeing=sum(f >= NEW for f in first_dis), control_shuffled_agreement=round(ctl / total, 5),
                         seconds=round(time.time() - t0, 1))
        ex.log(prec, out[prec])
    ex.result(**out, headline=f"HF argmax agreement: F16 {out['f16']['argmax_agreement']:.4f}, Q4_0 {out['q4_0']['argmax_agreement']:.4f}; "
                              f"shuffled control {out['f16']['control_shuffled_agreement']:.3f}")

"""EXP-030: build the neuron-SELECTION corpus (disjoint from every evaluation set).

Source tasks: original MBPP (Austin et al. 2021) 'full' train (374) + validation (90) splits. MBPP+ evaluation
problems come from the MBPP test/sanitized pool; we verify zero task_id overlap and near-zero text overlap here.
For each task we record, from the model itself (on-policy):
  * code_greedy: greedy solution; code_s1, code_s2: two samples (T=0.8, top_p=0.95, seeds 1, 2);
  * prose: a plain-English explanation of how to solve the task, explicitly without code.
Each code sample is executed against the original MBPP asserts -> pass/fail labels for correctness contrasts.
Generation uses the Q4_0 engine for speed (activations are later computed teacher-forced in bf16 PyTorch).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import DATA, SMOKE, bench  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.evaluate import render  # noqa: E402
from sne.execute import run_programs  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

PROSE = ("{text}\n\nExplain in plain English, in one or two short paragraphs, how you would solve this task. "
         "Do not write any code, code blocks, or function names.")
CODE = "{text}\nYour code should pass this test:\n{t0}\n\nRespond with the complete Python solution in a single ```python code block."

with Experiment("EXP-030", "selection-corpus-mbpp-train", category="localization", depends_on=["EXP-002"],
                motivation="Neuron/direction selection must use data disjoint from evaluation. Contrasting the model's "
                           "own code and its own prose about the *same* tasks isolates 'writing code' from 'thinking "
                           "about programming'. Pass/fail labels on its own samples give a correctness contrast.",
                hypothesis="After removing the 147 MBPP+ overlaps, the model passes roughly half of the remaining tasks greedily; sampling yields mixed "
                           "pass/fail pairs on a useful fraction of tasks.",
                config=dict(splits=["train", "validation"], samples=2, temperature=0.8, top_p=0.95,
                            engine="llama.cpp vulkan q4_0", parallel=8)) as ex:
    rows = []
    for sp in ["train", "validation"]:
        rows += [dict(r, split=sp) for r in map(json.loads, open(os.path.join(DATA, f"mbpp_full_{sp}.jsonl"),
                                                                 encoding="utf-8"))]
    # Leakage control. MBPP+ draws its 378 problems from the sanitized MBPP pool across ALL original splits, so
    # 147 of the 464 train/validation tasks are also MBPP+ evaluation problems. Drop every task whose id appears
    # anywhere in MBPP+ (dev or test half, or excluded), then verify no residual id or text overlap.
    plus = [json.loads(l) for l in open(os.path.join(DATA, "mbppplus.jsonl"), encoding="utf-8")]
    plus_ids = {str(p["task_id"]) for p in plus}
    n_before = len(rows)
    rows = [r for r in rows if str(r["task_id"]) not in plus_ids]
    ex.log(f"dropped {n_before - len(rows)} tasks shared with MBPP+; {len(rows)} remain")
    overlap_ids = sorted(str(r["task_id"]) for r in rows if str(r["task_id"]) in plus_ids)
    plus_text = {" ".join(p["prompt"].lower().split()) for p in plus}
    overlap_text = sum(" ".join(r["text"].lower().split()) in plus_text for r in rows)
    ex.log("residual task_id overlap with MBPP+:", overlap_ids, "exact text overlap:", overlap_text)
    if SMOKE:  # preflight: the whole pipeline on 6 tasks
        rows = rows[:6]
    tok = tokenizer()
    code_prompts = render(tok, [CODE.format(text=r["text"], t0=r["test_list"][0]) for r in rows])
    prose_prompts = render(tok, [PROSE.format(text=r["text"]) for r in rows])
    out = {}
    with LlamaServer(GGUF["q4_0"], parallel=8, log_path=os.path.join(ex.dir, "server.log")) as srv:
        g, st = srv.complete_many(code_prompts, max_tokens=8192)
        out["code_greedy"] = [x["text"] for x in g]
        ex.log("greedy", st)
        for s in (1, 2):
            g, st = srv.complete_many(code_prompts, max_tokens=8192,
                                      sampling=dict(temperature=0.8, top_p=0.95, top_k=0, seed=s))
            out[f"code_s{s}"] = [x["text"] for x in g]
            ex.log(f"sample {s}", st)
        g, st = srv.complete_many(prose_prompts, max_tokens=4096)
        out["prose"] = [x["text"] for x in g]
        ex.log("prose", st)
    labels = {}
    for k in ["code_greedy", "code_s1", "code_s2"]:
        jobs = []
        for i, (r, txt) in enumerate(zip(rows, out[k])):
            code = bench.extract_code(txt)
            prog = (r.get("test_setup_code") or "") + "\n" + code + "\n\n" + "\n".join(r["test_list"]) + "\n"
            jobs.append((str(i), prog))
        res = run_programs(jobs)
        labels[k] = [res[str(i)]["passed"] for i in range(len(rows))]
        ex.log(k, "pass rate", sum(labels[k]) / len(rows))
    prose_has_code = sum(("```" in p) or ("def " in p) for p in out["prose"])
    mixed = sum(len({labels["code_greedy"][i], labels["code_s1"][i], labels["code_s2"][i]}) == 2
                for i in range(len(rows)))
    corpus = [dict(task_id=r["task_id"], split=r["split"], text=r["text"], code_prompt=cp, prose_prompt=pp,
                   **{k: out[k][i] for k in out}, **{f"pass_{k}": labels[k][i] for k in labels})
              for i, (r, cp, pp) in enumerate(zip(rows, code_prompts, prose_prompts))]
    ex.save_json("corpus.json", corpus)
    ex.result(n_tasks=len(rows), overlap_task_ids=overlap_ids, overlap_exact_text=overlap_text,
              pass_rates={k: sum(v) / len(v) for k, v in labels.items()}, tasks_with_mixed_outcomes=mixed,
              prose_contaminated_with_code=prose_has_code,
              headline=f"greedy pass {sum(labels['code_greedy'])/len(rows):.3f}; mixed-outcome tasks {mixed}; "
                       f"id overlap {len(overlap_ids)}; prose-with-code {prose_has_code}")

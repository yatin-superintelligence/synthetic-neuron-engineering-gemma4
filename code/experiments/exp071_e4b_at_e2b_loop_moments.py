"""EXP-071: does E4B escape the moments where E2B loops, or does it simply never reach them?

EXP-013K ran E4B on the official stack: 0 of 58 tasks had a loop, against 16 of 58 for E2B at the same seed (EXP-044a), using one detector on
both. That leaves two explanations. E4B may escape the same states, which makes loops a property of the small model. Or its transcripts may
never reach such states, which makes the difference a matter of path. This run replays the exact saved E2B moments on E4B: same token
prompts, same thinking budget, same sampling. The tokenizers are byte-identical files. All loop moments are used, dev and test, plus 12
normal moments from the test split. No neuron edit is involved, and nothing here tunes the loop edit, so the held-out rule of EXP-066 stands.
Fixed before the run: the primary comparison is the test split at seeds 0 and 1 against E2B's rows from EXP-066 at the same seeds, Q4_0 with
Q4_0, F16 with F16, with a sign-flip test across tasks. Dev moments are reported as a rate only, because EXP-059 used other seeds.
The E4B F16 file is built from the QAT checkpoint on first use and removed after the full run to save disk.
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import MODELS, ROOT, SMOKE, loops as L  # noqa: E402
from sne.artifacts import LOCAL  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import tokenizer  # noqa: E402

SEEDS = (0, 1)
GG = os.path.join(MODELS, "gguf")
E4B = {"q4_0": os.path.join(GG, "gemma-4-E4B_q4_0-it.gguf"), "f16": os.path.join(GG, "e4b-qat-f16.gguf")}
SRC = os.path.join(MODELS, "gemma-4-E4B-it-qat-unquantized")
CONV = os.path.join(ROOT, "tools", "llama", "src", "llama.cpp-b12129", "convert_hf_to_gguf.py")
E2B_ROWS = os.path.join(ROOT, "experiments", "EXP-066_loop-edit-held-out")


def sign_flip(a, b, n_boot=20000, seed=71):
    """a and b map (case, seed) to a row. Returns the change in loops per task from a to b, with a sign-flip test across tasks."""
    by = {}
    for k in a:
        if k in b:
            t = k[0].split("/")[1]
            by[t] = by.get(t, 0) + int(a[k]["loop"]) - int(b[k]["loop"])
    nets = list(by.values())
    obs = sum(nets)
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in nets)) >= abs(obs) for _ in range(n_boot))
    return dict(tasks=len(nets), tasks_changed=sum(x != 0 for x in nets), net_loops_fewer_on_e4b=obs, sign_flip_p=round(hits / n_boot, 4),
                by_task={k: v for k, v in by.items() if v})


with Experiment("EXP-071", "e4b-at-e2b-loop-moments", category="diagnosis", depends_on=["EXP-013K", "EXP-044a", "EXP-066"],
                motivation="E4B shows no loops on the official stack. Whether it escapes the same states decides if loops are a small-model trait, and sets the simple baseline the loop edit must be compared with.",
                hypothesis="E4B loops far less than E2B from the same saved moments. If its rate matches E2B's, the official difference comes from the path taken and not from the model.",
                config=dict(splits=["dev", "test"], seeds=SEEDS, primary="test split, task-level sign-flip against EXP-066 base rows", early_exit=True, neuron_edit=None)) as ex:
    if not os.path.exists(E4B["f16"]):
        cmd = [sys.executable, CONV, SRC, "--outtype", "f16", "--outfile", E4B["f16"]]
        ex.log("CMD", " ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ex.log(p.stderr[-500:])
        if not os.path.exists(E4B["f16"]):
            raise RuntimeError("E4B F16 conversion failed")
    tok = tokenizer()
    cases = L.load()
    loops_only = [c for c in cases if c["kind"] == "loop"]
    controls = [c for c in L.load("test", "control")][:12]
    if SMOKE:
        loops_only, controls = loops_only[:2], controls[:1]
    ex.log("loop moments", len(loops_only), "tasks", len({c["instance_id"] for c in loops_only}), "controls", len(controls))
    if not loops_only:
        raise RuntimeError("no states to test")
    os.makedirs(os.path.join(LOCAL, ex.id), exist_ok=True)
    out = {}
    for prec in ("q4_0", "f16"):
        with LlamaServer(E4B[prec], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS, log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            summ, r, texts = L.run(srv, tok, loops_only + controls, seeds=SEEDS, early_exit=True)
        ex.save_json(f"rows_{prec}.json", r)
        json.dump(texts, open(os.path.join(LOCAL, ex.id, f"texts_{prec}.json"), "w", encoding="utf-8"))
        out[prec] = dict(all_moments=summ)
        ex.log(prec, summ)
        mine = {(x["case"], x["seed"]): x for x in r if x["kind"] == "loop"}
        split_of = {c["case"]: c["split"] for c in loops_only}
        for split in ("dev", "test"):
            part = [x for k, x in mine.items() if split_of.get(k[0]) == split]
            out[prec][f"{split}_loops"] = [sum(int(x["loop"]) for x in part), len(part)]
        p = os.path.join(E2B_ROWS, f"rows_{prec}_base.json")
        if os.path.exists(p):
            base = {(x["case"], x["seed"]): x for x in json.load(open(p, encoding="utf-8")) if x["kind"] == "loop"}
            out[prec]["test_against_e2b"] = dict(e2b_loops=[sum(int(x["loop"]) for x in base.values()), len(base)], **sign_flip(base, mine))
            ex.log(prec, out[prec]["test_against_e2b"])
        ex.result(arms=out)
    if not SMOKE:
        os.remove(E4B["f16"])
        ex.log("removed", E4B["f16"])
    ex.result(arms=out, headline=f"E4B at E2B loop moments: Q4_0 dev {out['q4_0']['dev_loops']}, test {out['q4_0']['test_loops']}; F16 dev {out['f16']['dev_loops']}, test {out['f16']['test_loops']} (loops, replays)")

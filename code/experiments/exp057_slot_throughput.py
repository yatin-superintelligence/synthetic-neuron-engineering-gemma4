"""EXP-057: how many parallel slots finish a batch of long agent contexts fastest?

Replays used 6 slots because short prompts gained from parallel decoding (EXP-001). On long contexts this was never tested:
one prompt alone runs near 300 tokens/s (EXP-055), yet six together averaged about 130 each, and generation fell from about 18
tokens/s to 5 per slot. This measures the real job: the same 6 real states, each generating 256 tokens, at 1, 2, 3, 4 and 6
slots, with total wall time as the measure, at both precisions. Two arms use one unified cache across slots (-kvu), because
six separate reserved caches may be what costs time. Physical batch is 512: a first attempt at 1024 lost the Vulkan device on
long contexts, most likely a GPU timeout, since one step grows with both batch size and context length. A crash at any
setting is recorded and the sweep moves on, because which settings are stable is part of the result.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import SMOKE, loops as L  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

ARMS = [(1, False), (2, False), (3, False), (4, False), (6, False), (3, True), (6, True)]
N, GEN, UB = 6, 256, 512

with Experiment("EXP-057", "slot-throughput-long-contexts", category="infrastructure", depends_on=["EXP-055"],
                motivation="Slot count was set on short prompts and carried over untested to long contexts, where every replay "
                           "now spends its time.",
                hypothesis="Fewer slots finish a batch of long contexts sooner, because attention cost grows with total cached tokens.",
                config=dict(arms=ARMS, states=N, gen_tokens=GEN, ubatch=UB)) as ex:
    tok = tokenizer()
    cases = [c for c in L.load("dev") if c["kind"] == "loop"][:2 if SMOKE else N]
    prompts = [L.render(tok, c) for c in cases]
    gen = 16 if SMOKE else GEN
    out = []
    for prec in ["q4_0", "f16"]:
        for np, unified in ARMS[:2] if SMOKE else ARMS:
            name = f"{prec}_np{np}" + ("_kvu" if unified else "")
            row = dict(precision=prec, slots=np, unified=unified)
            try:
                with LlamaServer(GGUF[prec], parallel=np, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS + (["-kvu"] if unified else []),
                                 log_path=os.path.join(ex.dir, f"server_{name}.log")) as srv:
                    t = time.time()
                    gens, st = srv.complete_many(prompts, max_tokens=gen, sampling=dict(temperature=0.2, top_p=0.95, top_k=0, seed=0, stop=L.STOP, cache_prompt=False), workers=np)
                    wall = time.time() - t
                row.update(wall_s=round(wall, 1), prompt_tokens=sum(g["n_prompt"] or 0 for g in gens), gen_tokens=st["gen_tokens"], s_per_state=round(wall / len(prompts), 1), status="ok")
            except Exception as e:  # noqa: BLE001
                row.update(status=repr(e)[:160])
            out.append(row)
            ex.log(row)
            ex.save_json("throughput.json", out)
    ok = [r for r in out if r["status"] == "ok"]
    best = {p: min((r for r in ok if r["precision"] == p), key=lambda r: r["wall_s"]) for p in ("q4_0", "f16")}
    ex.result(rows=out, headline="fastest: " + ", ".join(f"{p} {b['slots']} slots{' unified' if b['unified'] else ''}, {b['s_per_state']} s per state" for p, b in best.items()) +
                                 f"; at 6 slots {[r.get('s_per_state') for r in out if r['slots'] == 6 and not r['unified']]}; crashes {sum(r['status'] != 'ok' for r in out)}")

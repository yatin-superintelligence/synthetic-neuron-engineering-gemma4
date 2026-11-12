"""EXP-055: can long prompts be processed faster on the Arc GPU?

Long agent contexts take 1 to 3 minutes each at about 130 tokens/s, which is the main cost of every local replay. EXP-001
measured decoding on short prompts, so the prompt path has never been tuned. This sweeps the settings that could change it, on
one real 10000-token LoopBench prompt: logical and physical batch size, slot count, and context size. Each setting is one
prompt evaluation plus 32 generated tokens at both precisions, recording prompt tokens/s, generation tokens/s, package power
and iGPU power. Prompt cache is off, so each measure is a full evaluation. The first setting is repeated at the end to catch
drift from heat or background load.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import SMOKE, loops as L  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402
from sne.sysmon import snapshot  # noqa: E402

GRID = [dict(ub=512, b=2048, np=6), dict(ub=1024, b=2048, np=6), dict(ub=2048, b=2048, np=6), dict(ub=4096, b=4096, np=6),
        dict(ub=8192, b=8192, np=6), dict(ub=512, b=2048, np=1), dict(ub=2048, b=2048, np=1), dict(ub=512, b=2048, np=6)]

with Experiment("EXP-055", "prompt-speed-sweep", category="infrastructure", depends_on=["EXP-001"],
                motivation="I accepted 130 tokens/s on long prompts without testing it, then cut the states I test to fit. If the "
                           "rate can be raised, local replays need no such cut and the need for paid compute falls.",
                hypothesis="A larger physical batch raises prompt throughput, because the GPU is underused at 512 tokens per step.",
                config=dict(grid=GRID, target_prompt_tokens=10000)) as ex:
    tok = tokenizer()
    cases = L.load("dev")
    best = min(cases, key=lambda c: abs(len(tok(L.render(tok, c), add_special_tokens=False)["input_ids"]) - (1500 if SMOKE else 10000)))
    prompt = L.render(tok, best)
    n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    ex.log("prompt tokens", n)
    rows = []
    grid = GRID[:2] if SMOKE else GRID
    for prec in ["q4_0", "f16"]:
        for g in grid:
            args = list(L.SERVER_ARGS)
            cfg = dict(g, precision=prec)
            try:
                srv = LlamaServer(GGUF[prec], parallel=g["np"], ctx_per_slot=L.SLOT_CTX, extra_args=args + ["-b", str(g["b"]), "-ub", str(g["ub"])],
                                  log_path=os.path.join(ex.dir, f"server_{prec}_ub{g['ub']}_np{g['np']}.log"))
                with srv:
                    t = time.time()
                    r = srv.complete(prompt, 32, sampling=dict(temperature=0.2, top_p=0.95, seed=0, cache_prompt=False))
                    wall = time.time() - t
                    s = snapshot()
                tm = r["timings"]
                cfg.update(prompt_tps=round(tm.get("prompt_per_second", 0), 1), gen_tps=round(tm.get("predicted_per_second", 0), 2), wall_s=round(wall, 1),
                           n_prompt=r["n_prompt"], package_w=s.get("cpu_package_w"), igpu_w=s.get("igpu_w"), freq_mhz=s.get("cpu_freq_mhz"), status="ok")
            except Exception as e:  # noqa: BLE001
                cfg.update(status=repr(e)[:160])
            rows.append(cfg)
            ex.log(cfg)
            ex.save_json("sweep.json", rows)
    ok = [r for r in rows if r["status"] == "ok"]
    ex.result(rows=rows, headline="best prompt tokens/s: " + ", ".join(
        f"{p} {max((r for r in ok if r['precision'] == p), key=lambda r: r['prompt_tps'])['prompt_tps']} at ub {max((r for r in ok if r['precision'] == p), key=lambda r: r['prompt_tps'])['ub']}"
        for p in ("q4_0", "f16")) + f"; reference ub 512: {[r['prompt_tps'] for r in ok if r['ub'] == 512 and r['np'] == 6]}")

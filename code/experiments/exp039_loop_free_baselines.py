"""EXP-039: can a plain sampling setting remove thinking loops? Baselines every weight edit must beat (LoopBench dev).

All arms use settings the official stack accepts in configs/sampling.yaml, so each is a real zero-cost alternative to a
weight edit. Reference: T=0.2, top_p=0.95, thinking budget 4096, as shipped in the sample submission. Screens stop a
response once it already counts as a loop, since its label cannot change. Only the dev split is used, so the test split
stays clean for the final comparison.
Metrics: loop rate on loop contexts, loop rate on controls, tool-call rate on both. A setting that stops loops by
suppressing tool calls, or that hurts controls, is not a fix, so both are reported side by side.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import loops as L  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

SEEDS = (0, 1, 2)
ARMS = {  # name: (temperature, top_p, penalties, thinking budget, thinking on)
    "reference": (0.2, 0.95, {}, 4096, True),
    "freq_0.3": (0.2, 0.95, dict(frequency_penalty=0.3), 4096, True),
    "freq_0.6": (0.2, 0.95, dict(frequency_penalty=0.6), 4096, True),
    "pres_0.6": (0.2, 0.95, dict(presence_penalty=0.6), 4096, True),
    "t_0.8": (0.8, 0.95, {}, 4096, True),
    "budget_1024": (0.2, 0.95, {}, 1024, True),
    "thinking_off": (0.2, 0.95, {}, 0, False),
}


def no_think(tk, c):
    s = tk.apply_chat_template(c["messages"], tools=L.TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return s[len("<bos>"):] if s.startswith("<bos>") else s


with Experiment("EXP-039", "loop-free-baselines", category="baselines", depends_on=["EXP-038"],
                motivation="Reviewer failure to avoid: claiming a neural fix when a setting does the same job. The official "
                           "config allows penalties, temperature and the thinking budget, so these are the arms to beat.",
                hypothesis="Frequency penalties reduce loops but also reduce tool-call rate on controls, because tool calls "
                           "need repeated syntax tokens. Temperature alone helps little. A lower budget does not stop loops "
                           "but caps their cost.",
                config=dict(split="dev", seeds=SEEDS, slots=6, arms={k: list(v) for k, v in ARMS.items()}, early_exit=True)) as ex:
    tok = tokenizer()
    cases = L.load("dev")
    out = {}
    for prec in ["q4_0", "f16"]:
        out[prec] = {}
        with LlamaServer(GGUF[prec], parallel=6, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            for name, (t, p, pen, budget, think) in ARMS.items():
                # repeat_last_n=-1 counts the whole generated text, which is how vLLM applies these penalties
                summ, rows, _ = L.run(srv, tok, cases, seeds=SEEDS, temperature=t, top_p=p, extra=dict(pen, repeat_last_n=-1), budget=budget or 4096,
                                      thinking=think, early_exit=True, prompt_fn=None if think else no_think)
                out[prec][name] = summ
                ex.save_json(f"rows_{prec}_{name}.json", rows)
                ex.log(prec, name, summ)
                ex.result(arms=out)
    f = out["q4_0"]
    best = min((k for k in f if k != "reference"), key=lambda k: f[k]["loop_rate_on_loop_contexts"][0])
    ex.result(arms=out, headline=f"Q4_0 reference loop {f['reference']['loop_rate_on_loop_contexts'][0]:.2f}, control tool calls {f['reference']['tool_call_rate_on_controls'][0]:.2f}; "
                                 f"best {best}: loop {f[best]['loop_rate_on_loop_contexts'][0]:.2f}, control tool calls {f[best]['tool_call_rate_on_controls'][0]:.2f}")

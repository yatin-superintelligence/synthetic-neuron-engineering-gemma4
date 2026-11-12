"""Generic multi-arm evaluation used by most experiments.

An *arm* is one model condition: a base GGUF (precision arm) + optional LoRA adapter + optional system prompt.
Every experiment evaluates its arms at BOTH precisions (F16 full precision and Q4_0) unless stated otherwise,
saves every generation, and reports paired statistics against the reference arm of the same precision.
"""
import json
import os
from dataclasses import asdict, dataclass, field

from transformers import AutoTokenizer

from . import MODELS, SMOKE, bench
from .engine import LlamaServer
from .evaluate import mmlu_eval, run_repopatch, run_suite
from .stats import paired_compare

GGUF = {
    "f16": os.path.join(MODELS, "gguf", "e2b-qat-f16.gguf"),
    "q4_0": os.path.join(MODELS, "gguf", "gemma-4-E2B_q4_0-it.gguf"),
}
TOKENIZER = os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized")


@dataclass
class Arm:
    name: str
    precision: str = "f16"
    gguf: str = ""
    lora: str = ""          # path to LoRA GGUF ('' = none)
    lora_scale: float = 1.0
    system: str = ""
    extra_server_args: tuple = field(default_factory=tuple)
    thinking: bool = False

    def base(self):
        return self.gguf or GGUF[self.precision]


_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER)
    return _TOK


def load_suite(name):
    if name == "behavior":
        from .behavior import load_probes
        return load_probes()
    if name == "mmlu500":
        return bench.mmlu_sample(500)
    if name == "lcb":
        from . import lcb
        return lcb.load()
    if name.startswith("copybench"):
        from . import copybench
        return copybench.load(name.split("_")[1] if "_" in name else None)
    if name.startswith("repopatch"):
        from . import repobench
        return repobench.load_split(name.split("_")[1] if "_" in name else None)
    return bench.get_suite(name)


def evaluate_arms(ex, arms, suites, reference=None, parallel=8, max_tokens=None, sampling=None, ctx_per_slot=16384):
    """Returns {arm_name: {suite: summary}} and writes per-task outcomes + paired stats into the experiment dir."""
    tok = tokenizer()
    tasks = {s: load_suite(s) for s in suites}
    if SMOKE:  # a preflight checks the pipeline on a few tasks per suite; before 2026-11-08 it ran every suite in full (EXP-061)
        tasks = {s: (t[:6] if isinstance(t, list) else t) for s, t in tasks.items()}
    results, outcomes = {}, {}
    prior = {}
    rp = os.path.join(ex.dir, "results_resume.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            prior = json.load(f).get("results", {}).get("arms", {})
    behavior_rows = {}  # first arm evaluated per precision is the behavioural reference (put the base arm first)
    groups = {}
    for a in arms:
        groups.setdefault((a.base(), a.extra_server_args), []).append(a)
    for (base, extra), garms in groups.items():
        loras = sorted({a.lora for a in garms if a.lora})
        log = os.path.join(ex.dir, f"llama_server_{os.path.basename(base)}.log")
        with LlamaServer(base, loras=loras, parallel=parallel, ctx_per_slot=ctx_per_slot, extra_args=extra,
                         log_path=log) as srv:
            ex.log(f"server up for {os.path.basename(base)} with {len(loras)} adapters in {srv.load_seconds:.1f}s")
            for a in garms:
                scales = {loras.index(a.lora): a.lora_scale} if a.lora else None
                results[a.name], outcomes[a.name] = {}, {}
                for s in suites:
                    # resume: a finished arm/suite from an interrupted run of this experiment is reused, not recomputed
                    done_path = os.path.join(ex.dir, f"outputs_{a.name}_{s}.jsonl")
                    if os.path.exists(done_path) and s in prior.get(a.name, {}):
                        with open(done_path, encoding="utf-8") as f:
                            rows = [json.loads(l) for l in f]
                        results[a.name][s] = prior[a.name][s]
                        outcomes[a.name][s] = {r["task_id"]: r["passed"] for r in rows}
                        if s == "behavior" and a.precision not in behavior_rows:
                            behavior_rows[a.precision] = rows
                        ex.log(f"{a.name} {s}: resumed from saved outputs")
                        continue
                    if s == "behavior":
                        from .behavior import run_probes, summarize
                        rows, st = run_probes(srv, tok, lora_scales=scales, system=a.system or None)
                        ref = behavior_rows.get(a.precision)
                        summ = dict(bench="behavior", n=len(rows), **summarize(rows, ref), gen=st)
                        summ["pass_rate"], summ["passed"] = summ["check_rate"], summ["checks_passed"]
                        if ref is None:
                            behavior_rows[a.precision] = rows
                        for r in rows:
                            r["passed"] = bool(r["check"]) if r["check"] is not None else True
                            r["task_id"] = r["id"]
                    elif s.startswith("copybench"):
                        from . import copybench
                        summ, rows = copybench.run(srv, tok, tasks[s], system=a.system or None, lora_scales=scales)
                    elif s == "lcb":
                        from . import lcb
                        summ, rows = lcb.run(srv, tok, tasks[s], system=a.system or None, lora_scales=scales,
                                             max_tokens=max_tokens or 8192, thinking=a.thinking)
                    elif s == "mmlu500":
                        summ, rows = mmlu_eval(srv, tok, tasks[s], system=a.system or None, lora_scales=scales)
                    elif s.startswith("repopatch"):
                        summ, rows = run_repopatch(srv, tok, tasks[s], system=a.system or None, lora_scales=scales,
                                                   sampling=sampling)
                    else:
                        summ, rows = run_suite(srv, tok, tasks[s], system=a.system or None, lora_scales=scales,
                                               max_tokens=max_tokens, sampling=sampling, thinking=a.thinking)
                    results[a.name][s] = summ
                    outcomes[a.name][s] = {r["task_id"]: r["passed"] for r in rows}
                    with open(os.path.join(ex.dir, f"outputs_{a.name}_{s}.jsonl"), "w", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    ex.log(f"{a.name:28s} {s:13s} pass={summ['pass_rate']:.4f} ({summ['passed']}/{summ['n']}) "
                           f"cats={summ.get('categories', '')} sim={summ.get('mean_similarity_to_base', '')} "
                           f"tok/s={summ['gen']['tok_per_s']}")
                    ex.result(arms={k: v for k, v in results.items()})
    # paired comparisons vs the reference arm of the same precision
    comps = {}
    if reference:
        ref_by_prec = {a.precision: a.name for a in arms if a.name.startswith(reference)}
        for a in arms:
            r = ref_by_prec.get(a.precision)
            if not r or r == a.name:
                continue
            for s in suites:
                comps[f"{a.name} vs {r} [{s}]"] = paired_compare(outcomes[r][s], outcomes[a.name][s])
    ex.save_json("arms.json", [asdict(a) for a in arms])
    ex.result(arms=results, paired=comps)
    return results, comps

"""EXP-060: does the loop edit also change what the agent does after a failed edit, and does it harm copying?

Two questions, on the dev split, with the EXP-059 adapter:
1. Retries. After a failed edit, 141 of 209 retries resend the same old_string, and 1 of those applied. If the loop direction
   carries repetition in general, removing it should lower that rate, which would tie two separate failures to one circuit.
2. Harm. Editing requires copying text exactly. An edit that suppresses repetition could damage that, so the apply rate on normal
   edit states is measured too. A gain on loops that breaks edit_file would not be worth having.
Arms are base and edit at both precisions, one seed. A matched control follows only if there is an effect to explain.
Generated text goes to artifacts_local for reading.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, editbench as E, loops as L  # noqa: E402
from sne.artifacts import LOCAL  # noqa: E402
from sne.engine import LlamaServer  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.runner import GGUF, tokenizer  # noqa: E402

DIRS = {"exp036": os.path.join(ROOT, "artifacts_local", "kaggle", "exp029", "results", "traces"),
        "exp037": os.path.join(ROOT, "artifacts_local", "kaggle", "exp037", "results", "traces")}
N_RETRY, N_EDIT, MAX_PROMPT = 24, 16, 16000
ADAPTER = {p: os.path.join(LOCAL, "EXP-059", f"remove_loop_dirs_{p}.gguf") for p in ("q4_0", "f16")}

with Experiment("EXP-060", "loop-edit-on-edit-behaviour", category="improvement", depends_on=["EXP-059"],
                motivation="Perseveration and looping may share a cause. Exact copying is also what such an edit could damage.",
                hypothesis="The edit lowers same-text retries and leaves the apply rate on normal edit states unchanged.",
                config=dict(n_retry=N_RETRY, n_edit=N_EDIT, max_prompt=MAX_PROMPT, seeds=[0], split="dev")) as ex:
    tok = tokenizer()
    retries = [c for c in E.build_retries(DIRS) if c["split"] == "dev"]
    by = {}
    for c in retries:
        by.setdefault((c["run"], c["instance_id"]), []).append(c)
    retries = E.short_cases(tok, [c for k in sorted(by) for c in by[k][:3]], MAX_PROMPT)[:2 if SMOKE else N_RETRY]
    edits = E.short_cases(tok, E.load("dev", per_run_task=2), MAX_PROMPT)[:2 if SMOKE else N_EDIT]
    ex.log("retry states", len(retries), "edit states", len(edits))
    if not retries or not edits:
        raise RuntimeError("no states to test")
    os.makedirs(os.path.join(LOCAL, ex.id), exist_ok=True)
    out = {}
    for prec in ["q4_0", "f16"]:
        out[prec] = {}
        with LlamaServer(GGUF[prec], loras=[ADAPTER[prec]], parallel=L.SLOTS, ctx_per_slot=L.SLOT_CTX, extra_args=L.SERVER_ARGS,
                         log_path=os.path.join(ex.dir, f"server_{prec}.log")) as srv:
            for name, scales in (("base", {}), ("edit", {0: 1.0})):
                rows, texts, summ = E.retry_run(srv, tok, retries, lora_scales=scales)
                ex.save_json(f"rows_retry_{prec}_{name}.json", rows)
                json.dump(texts, open(os.path.join(LOCAL, ex.id, f"texts_retry_{prec}_{name}.json"), "w", encoding="utf-8"))
                es, erows = E.run(srv, tok, edits, lora_scales=scales)
                ex.save_json(f"rows_edit_{prec}_{name}.json", erows)
                out[prec][name] = dict(retry=summ, edit_states=es)
                ex.log(prec, name, summ, es["all"])
                ex.result(arms=out)
    a, b = out["f16"]["base"], out["f16"]["edit"]
    ex.result(arms=out, headline=f"F16 same-text retries {a['retry']['same_text']} to {b['retry']['same_text']} of {a['retry']['n']}; apply on edit states {a['edit_states']['all']['apply_rate']:.2f} to {b['edit_states']['all']['apply_rate']:.2f}")

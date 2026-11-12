"""EXP-093: edit-fidelity neurons for the E2B agent, training-free, on EXP-092's patch bench.

About half of E2B's edit_file calls fail to apply (EXP-044a: 51 of 110), and in failed-edit states its old_string is not in the
file at all in 34 of 42 replays (EXP-052). A wasted edit costs turns and context, and blind retries follow. This experiment builds
a neuron from E2B's own applying-minus-non-applying answers. It reuses EXP-092's machinery unchanged: the same bench, splits and
fit samples, its directions and neuron construction, and its scoring. The only change is the label.

Pre-registered (D-064, before any result):
  Label: an answer applies if its old_string occurs exactly once in the code shown (EXP-092's definition).
  Directions: E2B's 4 samples per fit state (saved by EXP-092). In states with both outcomes, the bf16 activations on tool-call
    tokens of applying minus non-applying answers, at layers 9 to 27 (every 3rd). The 2 layers with the best held-out pairwise
    accuracy are kept.
  Neuron: as in EXP-092 (read-out on tool-call tokens, calibrated write). The dev scan is the unedited model and alpha in
    {1, 3, 9} at both layers, selected by the most applying answers (ties: smaller alpha, earlier layer).
  Test: the unedited model, the selected neuron, and a control neuron (random orthogonal write, same calibrated shift).
  Gate for official agent runs: on test, applying answers beat the unedited model (exact McNemar, two-sided, p < 0.05) and the
    control neuron, and correct answers are not fewer than the unedited model's.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE  # noqa: E402
from sne.registry import Experiment  # noqa: E402

_src = open(os.path.join(os.path.dirname(__file__), "exp092_patch_neurons.py"), encoding="utf-8").read()
P = {"__file__": os.path.join(os.path.dirname(__file__), "exp092_patch_neurons.py"), "__name__": "exp092"}
exec(compile(_src.split('with Experiment("EXP-092"')[0], "exp092", "exec"), P)  # bench, scoring, Server, directions, save_neuron, mcnemar
B92 = os.path.join(ROOT, "artifacts_local", "EXP-092")
OUT = os.path.join(ROOT, "artifacts_local", "EXP-093")

with Experiment("EXP-093", "edit-fidelity-neurons", category="improvement", depends_on=["EXP-092"],
                motivation="Half of E2B's edits fail to apply, mostly because its old_string is not in the file; a neuron from its own applying-minus-"
                           "non-applying answers targets that step without training.",
                hypothesis="An edit-fidelity neuron raises the share of E2B's bench answers that apply against the unedited model (exact McNemar p < 0.05) "
                           "and against a control neuron, without fewer correct answers.",
                config=dict(label="applies", reuses="EXP-092 bench, splits, fit samples, neuron construction", alphas=P["ALPHAS"], layers=P["LAYERS"])) as ex:
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tasks, files = P["load_tasks"]()
    if SMOKE:
        tasks = tasks[:24]
    states, skipped = P["build_states"](tasks, ex)
    tok = AutoTokenizer.from_pretrained(P["official_model"]())
    rng = random.Random(92)  # EXP-092's split, reproduced exactly
    ids = sorted({s["iid"] for s in states})
    rng.shuffle(ids)
    n = len(ids)
    split = {i: ("fit" if k < n // 2 else "dev" if k < 3 * n // 4 else "test") for k, i in enumerate(ids)}
    for s in states:
        s["split"] = split[s["iid"]]

    def vprompt(prefix):
        p = P["G"]["render"](tok, prefix, True)
        return p[len("<bos>"):] if p.startswith("<bos>") else p
    prompts = {s["iid"]: vprompt(s["prefix"]) for s in states}
    base = json.load(open(os.path.join(B92, "base_greedy.json"), encoding="utf-8"))
    saved = json.load(open(os.path.join(B92, "fit_samples.json"), encoding="utf-8"))
    by_id = {s["iid"]: s for s in states}
    fit = [s for s in states if s["split"] == "fit"]
    samples = [(by_id[a["iid"]], dict(text=a["text"], correct=a["applies"])) for a in saved if a["iid"] in by_id]  # the label is 'applies'
    ex.log("states", len(states), "| fit samples", len(samples), "applying", sum(a["correct"] for _, a in samples),
           "| unedited applies on dev/test", {sp: sum(base[s["iid"]]["applies"] for s in states if s["split"] == sp) for sp in ("dev", "test")})
    vec, rep = P["directions"](tok, fit, samples, ex)
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()  # the HF model must leave the GPU before vLLM starts
    ex.save_json("directions.json", rep)
    sel = rep["selected_layers"]
    loras = {}
    for L in sel:
        for a in P["ALPHAS"]:
            loras[f"f_L{L}_a{a}"] = P["save_neuron"](os.path.join(OUT, f"f_L{L}_a{a}"), L, a * vec[L]["u1"], vec[L]["v"])
    dev = [s for s in states if s["split"] == "dev"]
    srv = P["Server"](loras, os.path.join(OUT, "vllm_scan.log"))
    try:
        scan = {nme: [P["score"](o[0], s) for s, o in zip(dev, srv.answers(nme, [prompts[s["iid"]] for s in dev]))] for nme in list(loras) + ["base"]}  # D-071
    finally:
        srv.close()
    counts = {k: dict(applies=sum(r["applies"] for r in v), correct=sum(r["correct"] for r in v)) for k, v in scan.items()}
    counts["base_stored"] = dict(applies=sum(base[s["iid"]]["applies"] for s in dev), correct=sum(base[s["iid"]]["correct"] for s in dev))
    ex.log("dev scan (of", len(dev), "):", counts)
    best = max(loras, key=lambda nme: (counts[nme]["applies"], -int(nme.split("_a")[1]), -sel.index(int(nme.split("_L")[1].split("_")[0]))))
    L, a = int(best.split("_L")[1].split("_")[0]), int(best.split("_a")[1])
    ctrl = P["save_neuron"](os.path.join(OUT, "control"), L, a * vec[L]["uc1"], vec[L]["v"])
    test = [s for s in states if s["split"] == "test"]
    srv = P["Server"]({"selected": loras[best], "control": ctrl}, os.path.join(OUT, "vllm_test.log"))
    try:
        res = {nme: [P["score"](o[0], s) for s, o in zip(test, srv.answers(nme, [prompts[s["iid"]] for s in test]))] for nme in ("base", "selected", "control")}
    finally:
        srv.close()
    ba = [r["applies"] for r in res["base"]]  # D-071: same-server unedited arm
    bc = [r["correct"] for r in res["base"]]
    sa, sc = [r["applies"] for r in res["selected"]], [r["correct"] for r in res["selected"]]
    ca, cc = [r["applies"] for r in res["control"]], [r["correct"] for r in res["control"]]
    x, y, p = P["mcnemar"](sa, ba)
    gate = p < 0.05 and x > y and sum(sa) > sum(ca) and sum(sc) >= sum(bc)
    ex.result(states=len(states), directions=rep, dev_scan=counts, selected=best,
              test=dict(n=len(test), applies=dict(base=sum(ba), selected=sum(sa), control=sum(ca), selected_only=x, base_only=y, p=round(p, 4)),
                        correct=dict(base=sum(bc), selected=sum(sc), control=sum(cc))), gate_passed=gate,
              headline=f"edit fidelity: test applies unedited {sum(ba)}/{len(test)}, {best} {sum(sa)} (control {sum(ca)}), McNemar p = {round(p, 4)}; "
                       f"correct {sum(bc)} -> {sum(sc)}; gate {'passed' if gate else 'not passed'}")

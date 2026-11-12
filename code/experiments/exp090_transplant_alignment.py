"""EXP-090 (numbered EXP-080 until 2026-11-08; renumbered because it never ran and EXP-080 now names the teacher neurons): can MLP neurons of a larger Gemma 4 (donor) be re-expressed in E2B's coordinates (recipient)?

Donor set by env SNE_DONOR (default 'e4b'). Steps (no gradient training anywhere):
  1. Layer correspondence by linear CKA of residual streams on 4k shared tokens (all layer pairs).
  2. For 6 recipient layers chosen where E2B code neurons concentrate (EXP-031), fit ridge maps on task half A:
     x_R -> x_D (MLP inputs) and r_D -> r_R (residuals). Report held-out R^2 on half B.
  3. Donor code neurons: donor activations on code vs prose tokens (Cohen's d), top 64 per matched donor layer.
  4. Read-side fidelity: Pearson r between the mapped neuron's activation in E2B and the donor neuron's activation, on
     held-out tokens. Controls: 64 random donor neurons per layer; and the true map applied with tokens shuffled
     between models (breaks correspondence, estimates chance-level correlation).
Outputs: CKA matrix, map quality, fidelity distributions, and the mapped neurons (published as a release asset).
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import MODELS, ROOT  # noqa: E402
from sne.artifacts import publish  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.transplant import capture_all, linear_cka, map_neurons, neuron_acts, ridge  # noqa: E402

DONOR = os.environ.get("SNE_DONOR", "e4b")
DONOR_DIR = {"e4b": "gemma-4-E4B-it-qat-unquantized", "12b": "gemma-4-12B-it-qat-unquantized"}[DONOR]
EXP_ID = {"e4b": "EXP-090", "12b": "EXP-092"}[DONOR]
E31 = glob.glob(os.path.join(ROOT, "experiments", "EXP-031_*"))[0]
CORPUS = json.load(open(glob.glob(os.path.join(ROOT, "experiments", "EXP-030_*", "corpus.json"))[0], encoding="utf-8"))
NFIT, NHOLD, TOPN = 100, 40, 64


def pairs(ids, kind):
    k, p = ("code_greedy", "code_prompt") if kind == "code" else ("prose", "prose_prompt")
    return [(CORPUS[i][p], CORPUS[i][k]) for i in ids]


with Experiment(EXP_ID, f"transplant-alignment-{DONOR}-to-e2b", category="transplant", depends_on=["EXP-031"],
                motivation="Before installing any donor neuron, measure whether the donor's features can be read in the "
                           "recipient's coordinates at all. Without high read-side fidelity a transplant is noise.",
                hypothesis="CKA finds a monotone depth correspondence. Ridge maps reach held-out R^2 > 0.6 for MLP inputs "
                           "at matched depths. Code-selective donor neurons transfer with median r > 0.5, above random "
                           "donor neurons and far above the shuffled-token control (~0).",
                config=dict(donor=DONOR, donor_dir=DONOR_DIR, n_fit_tasks=NFIT, n_hold_tasks=NHOLD, top_per_layer=TOPN,
                            ridge_lambda=1e-2, max_tokens_per_seq=192)) as ex:
    idx = list(range(len(CORPUS)))
    random.Random(80).shuffle(idx)
    A, B = idx[:NFIT], idx[NFIT:NFIT + NHOLD]
    per_layer = json.load(open(os.path.join(E31, "per_layer.json")))
    rec_layers = sorted(sorted(range(2, 33), key=lambda L: -per_layer[L]["n_code_d_gt1"])[:6])
    ex.log("recipient layers", rec_layers)
    from sne.model import load
    # ---- recipient pass
    mR, tok = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    nR = len(mR.model.language_model.layers)
    cka_R = capture_all(mR, tok, pairs(A[:20], "code") + pairs(A[:20], "prose"), list(range(nR)), what=("r",))
    fitR = capture_all(mR, tok, pairs(A, "code") + pairs(A, "prose"), rec_layers)
    holdR = capture_all(mR, tok, pairs(B, "code") + pairs(B, "prose"), rec_layers)
    nholdc = sum(min(192, len(tok(CORPUS[i]["code_greedy"] + "<turn|>", add_special_tokens=False)["input_ids"])) for i in B)
    wpostR = {L: mR.model.language_model.layers[L].post_feedforward_layernorm.weight.detach().float().cpu() for L in rec_layers}
    del mR
    import gc
    gc.collect()
    torch.xpu.empty_cache()
    # ---- donor pass
    mD, tokD = load(os.path.join(MODELS, DONOR_DIR))
    assert tokD.get_vocab() == tok.get_vocab(), "tokenizers differ"
    LD = mD.model.language_model.layers
    nD = len(LD)
    cka_D = capture_all(mD, tok, pairs(A[:20], "code") + pairs(A[:20], "prose"), list(range(nD)), what=("r",))
    cka = np.array([[linear_cka(cka_R[i]["r"], cka_D[j]["r"]) for j in range(nD)] for i in range(nR)])
    np.save(os.path.join(ex.dir, "cka.npy"), cka)
    match = {L: int(cka[L].argmax()) for L in rec_layers}
    ex.log("matched donor layers", match)
    don_layers = sorted(set(match.values()))
    fitD = capture_all(mD, tok, pairs(A, "code") + pairs(A, "prose"), don_layers)
    holdD = capture_all(mD, tok, pairs(B, "code") + pairs(B, "prose"), don_layers)
    W = {m: dict(g=LD[m].mlp.gate_proj.weight.detach().float().cpu(), u=LD[m].mlp.up_proj.weight.detach().float().cpu(),
                 d=LD[m].mlp.down_proj.weight.detach().float().cpu(),
                 wpost=LD[m].post_feedforward_layernorm.weight.detach().float().cpu()) for m in don_layers}
    del mD
    gc.collect()
    torch.xpu.empty_cache()
    results, mapped = {}, {}
    rng = np.random.default_rng(80)
    for L in rec_layers:
        m = match[L]
        Wx, r2x_fit = ridge(fitR[L]["x"], fitD[m]["x"])
        Wr, r2r_fit = ridge(fitD[m]["r"], fitR[L]["r"])
        hx = holdR[L]["x"].float() @ Wx
        r2x = 1 - float((hx - holdD[m]["x"].float()).pow(2).sum() / (holdD[m]["x"].float() - holdD[m]["x"].float().mean(0)).pow(2).sum())
        hr = holdD[m]["r"].float() @ Wr
        r2r = 1 - float((hr - holdR[L]["r"].float()).pow(2).sum() / (holdR[L]["r"].float() - holdR[L]["r"].float().mean(0)).pow(2).sum())
        # donor code selectivity on fit tokens (first half of fit tokens are code, second half prose)
        ncode_fit = sum(min(192, len(tok(CORPUS[i]["code_greedy"] + "<turn|>", add_special_tokens=False)["input_ids"])) for i in A)
        aD = neuron_acts(fitD[m]["x"], W[m]["g"], W[m]["u"])
        ac, ap = aD[:ncode_fit], aD[ncode_fit:]
        d = (ac.mean(0) - ap.mean(0)) / (0.5 * (ac.var(0) + ap.var(0))).sqrt().clamp_min(1e-6)
        top = torch.argsort(d, descending=True)[:TOPN]
        rnd = torch.tensor(rng.choice(W[m]["g"].shape[0], TOPN, replace=False))
        rmsD = float(fitD[m]["y"].float().pow(2).mean().sqrt())
        rmsR = float(fitR[L]["y"].float().pow(2).mean().sqrt())

        def fidelity(sel, shuffle=False):
            gR, uR, dR = map_neurons(W[m]["g"][sel], W[m]["u"][sel], W[m]["d"][:, sel], Wx, Wr, W[m]["wpost"],
                                     wpostR[L], rmsD, rmsR)
            xr = holdR[L]["x"]
            if shuffle:
                xr = xr[torch.randperm(xr.shape[0], generator=torch.Generator().manual_seed(0))]
            aR = neuron_acts(xr, gR, uR)
            aDh = neuron_acts(holdD[m]["x"], W[m]["g"][sel], W[m]["u"][sel])
            aR, aDh = aR - aR.mean(0), aDh - aDh.mean(0)
            r = (aR * aDh).sum(0) / (aR.norm(dim=0) * aDh.norm(dim=0)).clamp_min(1e-9)
            return r, (gR, uR, dR)
        r_top, neu = fidelity(top)
        r_rnd, neu_rnd = fidelity(rnd)
        r_shuf, _ = fidelity(top, shuffle=True)
        mapped[L] = dict(donor_layer=m, donor_idx=top.tolist(), donor_d=d[top].tolist(), g=neu[0], u=neu[1], d=neu[2],
                         fidelity=r_top.tolist(), rmsD=rmsD, rmsR=rmsR,
                         rand_idx=rnd.tolist(), rand_g=neu_rnd[0], rand_u=neu_rnd[1], rand_d=neu_rnd[2],
                         rand_fidelity=r_rnd.tolist())
        results[L] = dict(donor_layer=m, cka=float(cka[L, m]), r2_x_fit=r2x_fit, r2_x_hold=r2x, r2_r_fit=r2r_fit,
                          r2_r_hold=r2r, donor_top_d_median=float(d[top].median()),
                          fid_top_median=float(r_top.median()), fid_top_q25=float(r_top.quantile(0.25)),
                          fid_rand_median=float(r_rnd.median()), fid_shuffled_median=float(r_shuf.median()),
                          n_top_fid_gt05=int((r_top > 0.5).sum()))
        ex.log("layer", L, results[L])
    ex.save_json("alignment.json", results)
    os.makedirs(os.path.join(ROOT, "artifacts_local", ex.id), exist_ok=True)
    torch.save(mapped, os.path.join(ROOT, "artifacts_local", ex.id, "mapped_neurons.pt"))
    publish(ex, [os.path.join(ROOT, "artifacts_local", ex.id, "mapped_neurons.pt"), os.path.join(ex.dir, "cka.npy")],
            f"{ex.id}_mapped_neurons.zip")
    med = np.median([v["fid_top_median"] for v in results.values()])
    ex.result(alignment=results, headline=f"median read-fidelity of top donor code neurons {med:.3f}; random "
              f"{np.median([v['fid_rand_median'] for v in results.values()]):.3f}; shuffled "
              f"{np.median([v['fid_shuffled_median'] for v in results.values()]):.3f}; held-out R2(x) "
              f"{np.median([v['r2_x_hold'] for v in results.values()]):.3f}")

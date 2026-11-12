"""EXP-085: transplant MLP neurons from Qwen3.5-4B into Gemma 4 E2B, closed form, with writes fitted in E2B's own space.

Pre-registered in RESEARCH_LOG.md (2026-11-08 06:30 UTC). Donor: Qwen/Qwen3.5-4B (Apache-2.0; text model 32 layers, residual
2,560, MLP 9,216, SiLU gating). Recipient: Gemma 4 E2B (bf16 reference of the official W4A16 model; 35 layers, residual 1,536).
No gradient training anywhere: every map and weight below is a ridge or least-squares solution.

Data: Python source from one snapshot per task repository (data/kaggle/code_corpus_v1.jsonl, private). Fit on fastapi and
requests; check on rich, a repository the fit never sees. Text is cut into chunks of about 2,400 characters at line ends.

Steps, fixed before the run (changes after the smoke runs, before any loss result: the two recipient layers take neurons from
different Qwen layers; ridge strengths are chosen by cross-validation inside the fit repositories; chunks over 1,024 tokens are cut):
1. Alignment. Each model reads the same chunks separately. A position is usable when both tokenizers end a token at the same
   character offset; 24 such positions per chunk are sampled (seed 85), skipping the first 200 characters so each has context.
   At every layer the normalised MLP input is captured for both models, plus E2B's MLP output and final hidden state there.
2. Layer match. Linear CKA between every E2B layer and every Qwen layer on fit positions. The two E2B layers between 8 and 28
   with the highest best-match CKA receive neurons, each from its best-matching Qwen layer.
3. Read side. Ridge map W from E2B's MLP input to Qwen's at the matched layer. A Qwen neuron (gate g, up u, down d) is read in
   E2B as gate g' = W g / 1.702 and up u' = 1.702 W u, so that GELU-tanh(g'.x) (u'.x) ~ SiLU(g.x_q) (u.x_q), since
   GELU-tanh(z/1.702) ~ SiLU(z)/1.702. Read fidelity: held-out (rich) correlation between this and the Qwen neuron's own value.
4. Selection. Qwen neurons ranked by mean direct contribution to the correct next token on fit positions: activation times the
   neuron's write projected on (unembedding of the correct token minus the probability-weighted mean unembedding). Top K = 64
   per layer. Slots: the 64 E2B neurons per layer with the lowest mean |activation| times |down column| on fit positions.
5. Write side (the step that failed for E4B, EXP-077). The down weights of the installed neurons are fitted by ridge so that
   their activations predict, at fit positions, the MLP-output change that would raise E2B's own correct next token to first
   order (through post-FFN norm, layer scalar, final norm and the tied unembedding: target = rms_y rms_h / s * t / (w_post w_final),
   t = E[y] - p E). One global scale is then chosen on fit chunks from 0.25, 0.5, 1, 2 (the one with the lowest fit NLL).
6. Check. Mean next-token loss (NLL) on held-out rich chunks, for: unedited; transplant (top-K Qwen neurons); control A (K random
   Qwen neurons from the same layers, same read map, same fitted write and scale rule); control B (the slots cleared, nothing
   installed). Per-chunk paired differences with bootstrap 95% intervals. Gate for the next stage (MBPP, then agent runs): the
   transplant's held-out NLL must be below the unedited model's and below control A's, both intervals excluding zero.
Outputs: artifacts_local/EXP-085 (LoRA adapters for transplant and controls, rank 64 on gate, up and down at the two layers;
maps and indices); counts and losses in the record.
"""
import json
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # the full run ran out of memory on an L4 without it
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sne import DATA, MODELS, ROOT, SMOKE  # noqa: E402
from sne.model import device, empty_cache, layers, text_model  # noqa: E402
from sne.registry import Experiment  # noqa: E402
from sne.transplant import linear_cka, ridge  # noqa: E402

QWEN = "Qwen/Qwen3.5-4B"
CORPUS = os.path.join(DATA, "kaggle", "code_corpus_v1.jsonl")
OUT = os.path.join(ROOT, "artifacts_local", "EXP-085")
FIT_REPOS, TEST_REPO = ("fastapi", "requests"), "rich"
CHUNK, POS_PER_CHUNK, SKIP_CHARS = 2400, 24, 200
N_FIT, N_TEST = (12, 6) if SMOKE else (360, 120)
K, LAYER_RANGE, SCALES, GELU_K = (8 if SMOKE else 64), (8, 28), (0.25, 0.5, 1.0, 2.0), 1.702


def chunks(rng):
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    def cut(text):
        out, cur = [], ""
        for line in text.splitlines(keepends=True):
            if len(cur) + len(line) > CHUNK and len(cur) > SKIP_CHARS + 200:
                out.append(cur); cur = ""
            cur += line
        if len(cur) > SKIP_CHARS + 200:
            out.append(cur)
        return [c for c in out if len(c) <= 2 * CHUNK]  # a single very long line made 10k-token chunks (run 2 out of memory)
    fit = [(c, r["repo"]) for r in rows if r["repo"] in FIT_REPOS for c in cut(r["text"])]
    test = [c for r in rows if r["repo"] == TEST_REPO for c in cut(r["text"])]
    rng.shuffle(fit); rng.shuffle(test)
    fit = fit[:N_FIT]
    return [c for c, _ in fit], [r for _, r in fit], test[:N_TEST]


def ends(tok, text):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], [e for _, e in enc["offset_mapping"]]


def find_layers(model):
    """The language model's decoder layers: the module list whose length is the text config's layer count (a vision tower
    also has blocks with an mlp, so the first list found is not enough)."""
    cfg = getattr(model.config, "text_config", model.config)
    n = cfg.num_hidden_layers
    lm = getattr(getattr(model, "model", model), "language_model", None)
    if lm is not None and hasattr(lm, "layers") and len(lm.layers) == n:
        return lm.layers
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == n and hasattr(m[0], "mlp") and hasattr(m[0].mlp, "gate_proj"):
            return m
    raise RuntimeError("no decoder layers found")


@torch.no_grad()
def capture(model, tok, texts, positions_chars, dev, bos=None, want_targets=False, E=None):
    """Per text, at the token ending at each chosen character offset: normalised MLP input at every layer, plus (if asked)
    the next-token target direction t, the next token id, MLP-output rms per layer and final-hidden rms (E2B only)."""
    Ls = find_layers(model)
    st = {}
    hooks = [L.mlp.register_forward_pre_hook(lambda m, a, i=i: st.__setitem__(("x", i), a[0][0].float())) for i, L in enumerate(Ls)]
    if want_targets:
        hooks += [L.mlp.register_forward_hook(lambda m, a, o, i=i: st.__setitem__(("y", i), o[0].float())) for i, L in enumerate(Ls)]
        hooks.append(text_model(model).norm.register_forward_pre_hook(lambda m, a: st.__setitem__("h", a[0][0].float())))
    X = {i: [] for i in range(len(Ls))}
    extra = dict(t=[], y_id=[], rms_y={i: [] for i in range(len(Ls))}, rms_h=[])
    try:
        for text, offs in zip(texts, positions_chars):
            ids, e = ends(tok, text)
            idx = [e.index(o) for o in offs]
            full = ([bos] if bos is not None else []) + ids
            sh = 1 if bos is not None else 0
            pos = [i + sh for i in idx]
            out = model(input_ids=torch.tensor([full], device=dev), use_cache=False,
                        logits_to_keep=torch.tensor(pos, device=dev) if want_targets else 1)
            for i in X:
                X[i].append(st[("x", i)][pos].half().cpu())
            if want_targets:
                lg = out.logits[0].float()
                p = torch.softmax(lg, -1)
                y = torch.tensor([ids[i + 1] for i in idx], device=dev)
                extra["t"].append((E[y] - p @ E).half().cpu())
                extra["y_id"].append(y.cpu())
                for i in X:
                    extra["rms_y"][i].append(st[("y", i)][pos].pow(2).mean(-1).sqrt().cpu())
                extra["rms_h"].append(st["h"][pos].pow(2).mean(-1).sqrt().cpu())
            del out
            st.clear()
    finally:
        for h in hooks:
            h.remove()
    X = {i: torch.cat(v) for i, v in X.items()}
    if want_targets:
        extra = dict(t=torch.cat(extra["t"]), y_id=torch.cat(extra["y_id"]), rms_h=torch.cat(extra["rms_h"]),
                     rms_y={i: torch.cat(v) for i, v in extra["rms_y"].items()})
    return X, extra


@torch.no_grad()
def nll(model, tok, texts, dev, bos):
    """Mean next-token loss per chunk (tokens after the first 32)."""
    out = []
    for text in texts:
        ids = ([bos] + tok(text, add_special_tokens=False)["input_ids"])[:1025]
        x = torch.tensor([ids], device=dev)
        lg = model(input_ids=x, use_cache=False).logits[0, :-1].float()
        l = torch.nn.functional.cross_entropy(lg[32:], x[0, 33:], reduction="mean")
        out.append(float(l))
        del lg
    return np.array(out)


def gelu_tanh(z):
    return torch.nn.functional.gelu(z, approximate="tanh")


LAMS = (1e-2, 1e-1, 1.0, 10.0)


def ridge_cv(X, Y, a, b):
    """Ridge with its strength chosen by fitting on fit-repository a (fastapi) and scoring R^2 on fit-repository b (requests);
    refitted on all fit positions. The held-out repository (rich) is not used. Added 06:55 UTC, before any loss result: the
    fixed strength 0.01 gave read-map R^2 0.63 on fit positions and 0.05 to 0.07 on rich."""
    X, Y = X.float(), Y.float()
    best, best_r2 = LAMS[0], -1e9
    for lam in LAMS:
        W, _ = ridge(X[a], Y[a], lam)
        res = Y[b] - X[b] @ W
        r2 = 1 - float(res.pow(2).sum() / (Y[b] - Y[b].mean(0)).pow(2).sum())
        if r2 > best_r2:
            best, best_r2 = lam, r2
    W, r2 = ridge(X, Y, best)
    return W, r2, best, best_r2


def boot(d, rng, n=4000):
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 5), round(float(np.percentile(m, 97.5)), 5)]


with Experiment("EXP-085", "qwen-transplant", category="improvement", depends_on=["EXP-077"],
                motivation="Add capacity to E2B from another model family (Qwen3.5-4B), fixing the write side that failed for E4B (EXP-077).",
                hypothesis="Transplanted Qwen neurons with writes fitted in E2B's space lower held-out code loss below both the unedited model and random-Qwen-neuron controls.",
                config=dict(donor=QWEN, k=K, layer_range=LAYER_RANGE, scales=SCALES, chunk=CHUNK, positions_per_chunk=POS_PER_CHUNK,
                            fit_repos=FIT_REPOS, test_repo=TEST_REPO, n_fit=N_FIT, n_test=N_TEST)) as ex:
    torch.set_grad_enabled(False)  # nothing here is trained; weights read from the models must not carry gradients (smoke run 1 failed on it)
    rng = random.Random(85)
    fit_txt, fit_repo, test_txt = chunks(rng)
    ex.log("chunks", len(fit_txt), "fit", len(test_txt), "test")
    from transformers import AutoTokenizer
    dev = device()
    tq = AutoTokenizer.from_pretrained(QWEN)
    tg = AutoTokenizer.from_pretrained(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))

    def pick(texts):
        res = []
        for t in texts:
            _, eg = ends(tg, t)
            _, eq = ends(tq, t)
            common = sorted(set(eg[:-1]) & set(eq) - {o for o in eg if o < SKIP_CHARS})
            res.append(sorted(rng.sample(common, min(POS_PER_CHUNK, len(common)))))
        return res
    fit_pos, test_pos = pick(fit_txt), pick(test_txt)
    pos_repo = np.array([r for r, ps in zip(fit_repo, fit_pos) for _ in ps])
    cv_a, cv_b = np.where(pos_repo == "fastapi")[0], np.where(pos_repo == "requests")[0]
    ex.log("aligned positions", sum(map(len, fit_pos)), "fit", sum(map(len, test_pos)), "test")

    # ---- donor pass
    # the checkpoint is a vision-language model; its own class loads every weight under the names the file uses (a text-only
    # class could leave weights unloaded with only a warning). Loading is then checked: no missing or unexpected keys allowed.
    from transformers import AutoModelForImageTextToText
    qm, linfo = AutoModelForImageTextToText.from_pretrained(QWEN, dtype=torch.bfloat16, output_loading_info=True)
    bad = {k: v for k, v in linfo.items() if v and k in ("missing_keys", "mismatched_keys")}
    if bad:
        raise RuntimeError(f"donor weights did not load cleanly: { {k: list(v)[:5] for k, v in bad.items()} }")
    qm = qm.to(dev).eval()
    QL = find_layers(qm)
    Wq = {i: (L.mlp.gate_proj.weight.float().cpu(), L.mlp.up_proj.weight.float().cpu(), L.mlp.down_proj.weight.float().cpu()) for i, L in enumerate(QL)}
    Uq = qm.get_output_embeddings().weight  # kept in bf16 (a float copy is 2.5 GB, part of the L4 out-of-memory in run 1)
    Xq_fit, _ = capture(qm, tq, fit_txt, fit_pos, dev)
    Xq_test, _ = capture(qm, tq, test_txt, test_pos, dev)
    # donor next-token direction at fit positions, for ranking neurons
    tq_dir = []
    with torch.no_grad():
        for text, offs in zip(fit_txt, fit_pos):
            ids, e = ends(tq, text)
            idx = [e.index(o) for o in offs]
            lg = qm(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=torch.tensor(idx, device=dev)).logits[0].float()
            p = torch.softmax(lg, -1).to(Uq.dtype)
            y = torch.tensor([ids[i + 1] if i + 1 < len(ids) else ids[i] for i in idx], device=dev)
            tq_dir.append((Uq[y].float() - (p @ Uq).float()).half().cpu())
            del lg, p
    tq_dir = torch.cat(tq_dir).float()
    del qm, Uq
    empty_cache()
    ex.log("donor captured", {i: tuple(v.shape) for i, v in list(Xq_fit.items())[:1]})

    # ---- recipient pass
    from sne.model import load
    gm, _ = load(os.path.join(MODELS, "gemma-4-E2B-it-qat-unquantized"))
    GL = layers(gm)
    tm = text_model(gm)
    E = tm.embed_tokens.weight.float()
    bos = tg.bos_token_id
    Xg_fit, ext_fit = capture(gm, tg, fit_txt, fit_pos, dev, bos=bos, want_targets=True, E=E)
    Xg_test, _ = capture(gm, tg, test_txt, test_pos, dev, bos=bos)
    del E  # 1.6 GB of float embeddings, needed only for the targets (the smoke run hit an out-of-memory retry later)
    empty_cache()
    ex.log("recipient captured", tuple(Xg_fit[0].shape), "targets", tuple(ext_fit["t"].shape))

    # ---- layer match by CKA
    # linear CKA from centred Gram matrices on the GPU (identical to sne.transplant.linear_cka, far faster for 35 x 32 pairs)
    n_cka = min(3000, len(Xg_fit[0]))
    def gram(X):
        X = X[:n_cka].float().to(dev)
        X = X - X.mean(0)
        K = X @ X.T
        return K / K.norm()
    Gq = {j: gram(Xq_fit[j]) for j in Xq_fit}
    cka = np.zeros((len(Xg_fit), len(Xq_fit)))
    for i in range(LAYER_RANGE[0], LAYER_RANGE[1] + 1):
        Ki = gram(Xg_fit[i])
        for j in Xq_fit:
            cka[i, j] = float((Ki * Gq[j]).sum())
        del Ki
    del Gq
    empty_cache()
    best = {i: int(cka[i].argmax()) for i in range(LAYER_RANGE[0], LAYER_RANGE[1] + 1)}
    # two recipient layers with different donor layers (changed 06:28 UTC after the smoke run, before any real result: in the
    # smoke run both top recipient layers matched Qwen layer 30, which would have installed the same 64 neurons twice)
    order = sorted(best, key=lambda i: -cka[i, best[i]])
    chosen = [order[0]] + [next(i for i in order[1:] if best[i] != best[order[0]])]
    ex.log("cka best per recipient layer", {i: (best[i], round(cka[i, best[i]], 3)) for i in best}, "chosen", [(i, best[i]) for i in chosen])

    w_final = tm.norm.weight.float()
    rec = dict(layers={}, nll={})
    edits = {"transplant": {}, "control_random": {}, "control_clear": {}}
    nrng = np.random.default_rng(85)
    for l in chosen:
        m = best[l]
        gq, uq, dq = Wq[m]
        W, r2, lam_read, r2_cv = ridge_cv(Xg_fit[l], Xq_fit[m], cv_a, cv_b)  # x_g W ~ x_q
        res_t = Xq_test[m].float() - Xg_test[l].float() @ W
        r2_test = 1 - float(res_t.pow(2).sum() / (Xq_test[m].float() - Xq_test[m].float().mean(0)).pow(2).sum())
        # donor activations and ranking
        aq_fit = torch.nn.functional.silu(Xq_fit[m].float() @ gq.T) * (Xq_fit[m].float() @ uq.T)
        proj = tq_dir @ dq  # [pos, 9216]: correct-token direction projected on each neuron's write
        score = (aq_fit * proj).mean(0)
        top = torch.topk(score, K).indices.tolist()
        rnd = nrng.choice(dq.shape[1], K, replace=False).tolist()
        # recipient slots: least used
        Gg, Ug, Dg = (GL[l].mlp.gate_proj.weight.float(), GL[l].mlp.up_proj.weight.float(), GL[l].mlp.down_proj.weight.float())
        xg = Xg_fit[l].float().to(dev)
        ag = gelu_tanh(xg @ Gg.T) * (xg @ Ug.T)
        use = ag.abs().mean(0) * Dg.norm(dim=0)
        slots = torch.topk(-use, K).indices.cpu().tolist()
        # write target in MLP-output space (first order, as in sne.conditional.write_vector)
        w_post = GL[l].post_feedforward_layernorm.weight.float().cpu()
        s = float(GL[l].layer_scalar.float().cpu())
        target = (ext_fit["rms_y"][l][:, None] * ext_fit["rms_h"][:, None] / s) * ext_fit["t"].float() / (w_post * w_final.cpu())[None]
        info = dict(donor_layer=m, cka=round(float(cka[l, m]), 3), read_map_ridge=lam_read, read_map_r2_fit=round(float(r2), 3),
                    read_map_r2_cv_requests=round(r2_cv, 3), read_map_r2_heldout=round(r2_test, 3))
        for name, neurons in (("transplant", top), ("control_random", rnd)):
            g_new = (W @ gq[neurons].T).T / GELU_K   # [K, 1536]
            u_new = (W @ uq[neurons].T).T * GELU_K
            A_fit = gelu_tanh(Xg_fit[l].float() @ g_new.T) * (Xg_fit[l].float() @ u_new.T)
            A_test = gelu_tanh(Xg_test[l].float() @ g_new.T) * (Xg_test[l].float() @ u_new.T)
            aq_test = torch.nn.functional.silu(Xq_test[m].float() @ gq[neurons].T) * (Xq_test[m].float() @ uq[neurons].T)
            fid = [float(np.corrcoef(A_test[:, k].numpy(), aq_test[:, k].numpy())[0, 1]) for k in range(len(neurons))]
            D_new, wr2, lam_w, wr2_cv = ridge_cv(A_fit, target, cv_a, cv_b)  # [K, 1536]
            edits[name][l] = dict(slots=slots, g=g_new, u=u_new, d=D_new.T)  # down columns [1536, K]
            info[name] = dict(read_fidelity_median=round(float(np.nanmedian(fid)), 3), read_fidelity_p25=round(float(np.nanpercentile(fid, 25)), 3),
                              write_fit_r2=round(float(wr2), 4), write_ridge=lam_w, write_r2_cv=round(wr2_cv, 4), neurons=neurons[:16])
        edits["control_clear"][l] = dict(slots=slots, g=torch.zeros(K, Gg.shape[1]), u=torch.zeros(K, Gg.shape[1]), d=torch.zeros(Dg.shape[0], K))
        rec["layers"][l] = info
        ex.log("layer", l, info)
        del ag, xg
        empty_cache()

    # ---- apply, measure, restore
    orig = {l: (GL[l].mlp.gate_proj.weight.data.clone(), GL[l].mlp.up_proj.weight.data.clone(), GL[l].mlp.down_proj.weight.data.clone()) for l in chosen}

    def apply(name, scale):
        for l, e in edits[name].items():
            G, U, D = (x.clone() for x in orig[l])
            sl = torch.tensor(e["slots"], device=G.device)
            G[sl] = e["g"].to(G.device, G.dtype)
            U[sl] = e["u"].to(U.device, U.dtype)
            D[:, sl] = (scale * e["d"]).to(D.device, D.dtype)
            GL[l].mlp.gate_proj.weight.data.copy_(G); GL[l].mlp.up_proj.weight.data.copy_(U); GL[l].mlp.down_proj.weight.data.copy_(D)

    def restore():
        for l in chosen:
            for mod, w in zip((GL[l].mlp.gate_proj, GL[l].mlp.up_proj, GL[l].mlp.down_proj), orig[l]):
                mod.weight.data.copy_(w)

    fit_eval = fit_txt[: max(4, len(fit_txt) // 6)]
    base_fit, base_test = nll(gm, tg, fit_eval, dev, bos), nll(gm, tg, test_txt, dev, bos)
    rec["nll"]["unedited"] = dict(fit=round(float(base_fit.mean()), 5), test=round(float(base_test.mean()), 5))
    nrng2 = np.random.default_rng(850)
    for name in ("transplant", "control_random", "control_clear"):
        scales = SCALES if name != "control_clear" else (1.0,)
        fits = {}
        for sc in scales:
            apply(name, sc)
            fits[sc] = float(nll(gm, tg, fit_eval, dev, bos).mean())
            restore()
        sc = min(fits, key=fits.get)
        apply(name, sc)
        t = nll(gm, tg, test_txt, dev, bos)
        restore()
        d = t - base_test
        rec["nll"][name] = dict(scale=sc, fit_by_scale={str(k): round(v, 5) for k, v in fits.items()}, test=round(float(t.mean()), 5),
                                test_minus_unedited=round(float(d.mean()), 5), ci95=boot(d, nrng2))
        rec["nll"][name]["_per_chunk"] = t
        ex.log("nll", name, {k: v for k, v in rec["nll"][name].items() if k != "_per_chunk"})
    tr, cr = rec["nll"]["transplant"].pop("_per_chunk"), rec["nll"]["control_random"].pop("_per_chunk")
    rec["nll"]["control_clear"].pop("_per_chunk")
    dtc = tr - cr
    rec["nll"]["transplant_minus_control_random"] = dict(mean=round(float(dtc.mean()), 5), ci95=boot(dtc, nrng2))
    gate = rec["nll"]["transplant"]["ci95"][1] < 0 and rec["nll"]["transplant_minus_control_random"]["ci95"][1] < 0
    rec["gate_passed"] = bool(gate)

    # ---- adapters (rank K on gate, up, down at the chosen layers; lora_alpha = K so the scale is 1)
    from safetensors.torch import save_file
    for name in edits:
        sc = rec["nll"][name]["scale"]
        tensors = {}
        for l, e in edits[name].items():
            G0, U0, D0 = (x.float().cpu() for x in orig[l])
            sl = e["slots"]
            onehot = torch.zeros(G0.shape[0], len(sl))
            onehot[sl, torch.arange(len(sl))] = 1.0
            base = f"base_model.model.model.language_model.layers.{l}.mlp"
            tensors[f"{base}.gate_proj.lora_A.weight"] = (e["g"] - G0[sl]).contiguous()
            tensors[f"{base}.gate_proj.lora_B.weight"] = onehot.clone().contiguous()
            tensors[f"{base}.up_proj.lora_A.weight"] = (e["u"] - U0[sl]).contiguous()
            tensors[f"{base}.up_proj.lora_B.weight"] = onehot.clone().contiguous()
            tensors[f"{base}.down_proj.lora_A.weight"] = onehot.T.clone().contiguous()
            tensors[f"{base}.down_proj.lora_B.weight"] = (sc * e["d"] - D0[:, sl]).contiguous()
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
        json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=K, lora_alpha=K,
                       target_modules=["gate_proj", "up_proj", "down_proj"], layers_to_transform=chosen, lora_dropout=0.0, bias="none",
                       fan_in_fan_out=False, init_lora_weights=True, inference_mode=True, use_rslora=False),
                  open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"), indent=2)
    json.dump(dict(cka=cka.round(4).tolist(), chosen=chosen, best=best), open(os.path.join(OUT, "alignment.json"), "w"), indent=1)
    ex.result(**rec, headline=f"held-out code NLL: unedited {rec['nll']['unedited']['test']}, transplant {rec['nll']['transplant']['test']} "
                              f"(diff {rec['nll']['transplant']['test_minus_unedited']} {rec['nll']['transplant']['ci95']}), random-Qwen control "
                              f"{rec['nll']['control_random']['test']}, cleared slots {rec['nll']['control_clear']['test']}; gate {'passed' if gate else 'not passed'}")

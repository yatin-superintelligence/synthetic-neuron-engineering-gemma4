"""EXP-086: more layers for E2B without training. Duplicate one block of E2B's own layers and test whether coding improves.

The author's brief asks for more neurons and layers in the smallest Gemma. Neurons were added and transplanted (EXP-076 to EXP-085);
layers were not (idea I6 in RESEARCH_LOG.md, open since 2026-11-06). This experiment adds layers: a block of k consecutive decoder
layers is run twice (layers 0 .. s+k-1, then s .. s+k-1 again, then the rest), giving 35 + k layers with shared weights, as in
layer-repetition ("self-merge") models. Nothing is trained.

Gemma 4 E2B constraints, read from transformers' modeling_gemma4.py and the checkpoint (2026-11-08):
  layers 0 to 14 compute their own keys and values; layers 15 to 34 (num_kv_shared_layers = 20) reuse those of the last
  non-shared layer of their attention type (13 for sliding, 14 for full) and have no k_proj or v_proj weights. Blocks are therefore
  taken inside 1..12 or inside 15..34, so the layers that store the shared keys and values stay the last non-shared ones.
  Each layer i also takes slice i of a per-layer embedding: embed_tokens_per_layer [262144, 35*256] and
  per_layer_model_projection [35*256, 1536]. A duplicated layer gets a copy of its own slice.
The new model is a standard Gemma 4 checkpoint (num_hidden_layers 35 + k, layer_types extended, num_kv_shared_layers 20 or 20 + k),
built by copying tensors of the official W4A16 checkpoint, and is served by the official vLLM 0.19.1 like the unedited model.

Pre-registered (D-056, before any result):
  Scan, MBPP+ dev half (189 problems, split seed 20261105), greedy, no thinking, up to 8192 new tokens, the official W4A16 model:
    pipeline check: the official folder and an identity rebuild (same tensors through the same code) must agree within 3 problems,
    or the job stops;
    21 blocks: k = 2 at s = 1, 3, 5, 7, 9, 11 and 15, 17, ..., 33; k = 4 at s = 3, 7, 17, 23, 29.
  Selection: the block with the most dev passes (ties: smaller k, then smaller s).
  Confirmation, MBPP+ test half (189), never used for selection: the selected block, the unedited model, and a control block (a
    block of the same k drawn at random with seed 86, not the selected one), at W4A16; then the unedited model and the selected
    block at bf16 (the public QAT unquantized checkpoint) for the precision check.
  Deviation D-057 (2026-11-08 13:34 UTC, before any block's pass count was seen): the scan caps answers at 4,096 new tokens
    instead of 8,192. In the first two dev runs no passing answer was longer than 2,150 tokens and every answer over 3,072 tokens
    ran past 8,000 (loops), so the cap changes no pass of the unedited model and stops looping answers early; looping blocks had
    made one configuration take about 15 min. Configurations already run at 8,192 are scored under the cap from their saved rows
    (a pass counts only if its answer is at most 4,096 tokens). The confirmation on the test half keeps 8,192.
  Gate for official agent runs: the selected block beats the unedited model on the test half at W4A16 (exact McNemar, two-sided,
    p < 0.05) and is not lower at bf16. Otherwise the scan is the result.
  The main competition accepts only the official checkpoints plus LoRA adapters, so a deeper model is a paper result only.
"""
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE, bench  # noqa: E402
from sne.evaluate import run_suite  # noqa: E402
from sne.registry import Experiment  # noqa: E402

N_LAYERS, N_SHARED, PLE = 35, 20, 256
BLOCKS = [(s, 2) for s in (1, 3, 5, 7, 9, 11)] + [(s, 2) for s in range(15, 34, 2)] + [(3, 4), (7, 4), (17, 4), (23, 4), (29, 4)]
LAYER = re.compile(r"^(.*language_model\.(?:model\.)?layers\.)(\d+)(\..+)$")
STACKED = ("embed_tokens_per_layer", "per_layer_model_projection")  # tensors with one 256-wide slice per layer
WORK = "/content/dup"
OUT = os.path.join(ROOT, "artifacts_local", "EXP-086")
PORT = 8011
SCAN_MAX = 4096  # D-057


def official_dir():
    for c in sorted(glob.glob("/content/kaggle_input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/*")):
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    raise FileNotFoundError("official E2B W4A16 checkpoint not found under /content/kaggle_input")


def order_for(s, k):
    if s is None:
        return list(range(N_LAYERS))
    assert (1 <= s and s + k <= 13) or (s >= 15 and s + k <= N_LAYERS), (s, k)
    return list(range(s + k)) + list(range(s, s + k)) + list(range(s + k, N_LAYERS))


def new_config(cfg, order, shared_block):
    cfg = json.loads(json.dumps(cfg))
    t = cfg.get("text_config", cfg)
    k = len(order) - N_LAYERS
    t["layer_types"] = [t["layer_types"][j] for j in order]
    t["num_hidden_layers"] = len(order)
    t["num_kv_shared_layers"] = N_SHARED + (k if shared_block else 0)
    q = cfg.get("quantization_config")
    if q and isinstance(q.get("ignore"), list):  # module names tied to a layer index are renamed for every copy of that layer
        out = []
        for name in q["ignore"]:
            m = LAYER.match(name)
            out += [f"{m.group(1)}{p}{m.group(3)}" for p, j in enumerate(order) if str(j) == m.group(2)] if m else [name]
        q["ignore"] = out
    return cfg


def stacked(name, t, order):
    """Reorders a per-layer stacked tensor (or its quantization metadata) so slice p holds layer order[p]'s slice."""
    import torch
    full = N_LAYERS * PLE
    if name.endswith("weight_shape"):
        v = t.clone()
        v[v == full] = len(order) * PLE
        return v
    dims = [i for i, d in enumerate(t.shape) if d == full]
    if len(dims) != 1:
        raise RuntimeError(f"unexpected layout for {name} {tuple(t.shape)} {t.dtype}: no single axis of size {full}")
    d = dims[0]
    idx = torch.tensor(order)
    return t.unflatten(d, (N_LAYERS, PLE)).index_select(d, idx).flatten(d, d + 1).contiguous()


def build(src, dst, order, shared_block, dry=False):
    """Writes a checkpoint whose decoder runs the layers in `order`. With dry=True only names and shapes are checked."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not dry:
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        for f in os.listdir(src):
            p = os.path.join(src, f)
            if os.path.isfile(p) and not f.endswith(".safetensors") and f not in ("config.json", "model.safetensors.index.json"):
                shutil.copy2(p, dst)
        json.dump(new_config(cfg, order, shared_block), open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=2)
    wmap, report = {}, dict(tensors_in=0, tensors_out=0, stacked=[])
    for f in files:
        out = {}
        with safe_open(f, "pt") as s:
            for name in s.keys():
                report["tensors_in"] += 1
                m = LAYER.match(name)
                if m and "language_model" in name:
                    new = [f"{m.group(1)}{p}{m.group(3)}" for p, j in enumerate(order) if str(j) == m.group(2)]
                    if dry:
                        report["tensors_out"] += len(new)
                        continue
                    t = s.get_tensor(name)
                    for n in new:
                        out[n] = t.clone() if len(new) > 1 else t
                elif "language_model" in name and any(x in name for x in STACKED):
                    if dry:
                        shape = s.get_slice(name).get_shape()
                        report["stacked"].append((name, shape))
                        report["tensors_out"] += 1
                        continue
                    out[name] = stacked(name, s.get_tensor(name), order)
                    report["stacked"].append((name, list(out[name].shape)))
                else:
                    if dry:
                        report["tensors_out"] += 1
                        continue
                    out[name] = s.get_tensor(name)
        if not dry:
            report["tensors_out"] += len(out)
            save_file(out, os.path.join(dst, os.path.basename(f)), metadata={"format": "pt"})
            wmap.update({n: os.path.basename(f) for n in out})
            del out
    if not dry and len(files) > 1:
        total = sum(os.path.getsize(os.path.join(dst, os.path.basename(f))) for f in files)
        json.dump(dict(metadata=dict(total_size=total), weight_map=wmap), open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
    return report


class Vllm:
    """The official vLLM, one model at a time, with the same completion interface as sne.engine.LlamaServer."""

    def __init__(self, model_dir, log_path):
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir, "--served-model-name", "m",
               "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "16384", "--gpu-memory-utilization", "0.88",
               "--enforce-eager", "--max-num-seqs", "192", "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]
        self.log_path = log_path
        self.p = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 1200:
            if self.p.poll() is not None:
                raise RuntimeError("vLLM exited while loading: " + open(log_path, encoding="utf-8", errors="replace").read()[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                self.load_s = round(time.time() - t0, 1)
                return
            except Exception:  # noqa: BLE001
                time.sleep(5)
        self.close()
        raise RuntimeError("vLLM did not become ready in 20 min")

    def _one(self, prompt, max_tokens):
        body = json.dumps(dict(model="m", prompt=prompt, max_tokens=max_tokens, temperature=0.0)).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        c = r["choices"][0]
        return dict(text=c["text"], n_gen=r["usage"]["completion_tokens"], n_prompt=r["usage"]["prompt_tokens"],
                    stop_type="limit" if c.get("finish_reason") == "length" else "stop")

    def complete_many(self, prompts, max_tokens, lora_scales=None, sampling=None):
        t0 = time.time()
        with ThreadPoolExecutor(192) as ex:
            gens = list(ex.map(lambda p: self._one(p, max_tokens), prompts))
        wall = time.time() - t0
        n = sum(g["n_gen"] for g in gens)
        return gens, dict(wall_s=round(wall, 1), gen_tokens=n, tok_per_s=round(n / max(wall, 1e-6), 1))

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        time.sleep(5)


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired pass lists a, b: returns (a only, b only, p)."""
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    n = x + y
    if n == 0:
        return x, y, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(x, y) + 1)) / 2 ** n
    return x, y, min(1.0, 2 * tail)


def capped(rows, cap):
    """Scores saved rows as if answers had been cut at `cap` new tokens: a pass needs an answer of at most cap tokens."""
    out = []
    for r in rows:
        r = dict(r)
        if (r.get("n_gen") or 0) > cap:
            r["passed"], r["category"] = False, "over_cap"
        out.append(r)
    n, k = len(out), sum(r["passed"] for r in out)
    cats = {}
    for r in out:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    return dict(n=n, passed=k, pass_rate=round(k / n, 4), categories=cats, scored_from_saved_rows=True, cap=cap,
                mean_gen_tokens=round(sum(min(r["n_gen"] or 0, cap) for r in out) / n, 1), load_s=None, total_s=None), out


def evaluate(model_dir, tag, tasks, tok, ex, max_tokens=None):
    saved = os.path.join(OUT, f"rows_{tag}.json")
    if max_tokens and os.path.exists(saved):  # a dev configuration already run (at 8,192) before the restart: scored under the cap
        summ, rows = capped(json.load(open(saved, encoding="utf-8")), max_tokens)
        ex.log(f"[dup] {tag}: {summ['passed']}/{summ['n']} passed (from saved rows, cap {max_tokens}) | categories {summ['categories']}")
        return summ, {r["task_id"]: r["passed"] for r in rows}
    srv = Vllm(model_dir, os.path.join(OUT, f"vllm_{tag}.log"))
    try:
        summ, rows = run_suite(srv, tok, tasks, max_tokens=max_tokens)
    finally:
        srv.close()
    summ["load_s"] = srv.load_s
    json.dump(rows, open(saved, "w", encoding="utf-8"))
    ex.log(f"[dup] {tag}: {summ['passed']}/{summ['n']} passed | categories {summ['categories']} | mean new tokens {summ['mean_gen_tokens']}"
           f" | load {summ['load_s']} s, total {summ['total_s']} s")
    return summ, {r["task_id"]: r["passed"] for r in rows}


if __name__ == "__main__" and "--dry" in sys.argv:  # local check of names and shapes on the bf16 checkpoint, no tensors loaded
    src = sys.argv[sys.argv.index("--dry") + 1]
    for s, k in [(None, 0), (3, 4), (17, 4)]:
        o = order_for(s, k)
        r = build(src, None, o, s is not None and s >= 15, dry=True)
        c = new_config(json.load(open(os.path.join(src, "config.json"))), o, s is not None and s >= 15)
        t = c.get("text_config", c)
        print(s, k, "layers", t["num_hidden_layers"], "shared", t["num_kv_shared_layers"], "first shared", t["num_hidden_layers"] - t["num_kv_shared_layers"],
              "tensors", r["tensors_in"], "->", r["tensors_out"], "stacked", r["stacked"])
    sys.exit(0)

with Experiment("EXP-086", "layer-duplication", category="improvement", depends_on=["EXP-044a"],
                motivation="The brief asks for more layers as well as more neurons. Re-running a block of E2B's own layers is the training-free way "
                           "to add depth; it has not been tested here (idea I6).",
                hypothesis="Some block, run twice, raises E2B's MBPP+ pass count on the held-out test half against the unedited model (exact McNemar "
                           "p < 0.05) and against a control block of the same size; most blocks lower it.",
                config=dict(blocks=BLOCKS, scan="MBPP+ dev 189, answers capped at 4,096 new tokens (D-057)", confirm="MBPP+ test 189, W4A16 and bf16", max_tokens=8192, thinking=False,
                            engine="vLLM 0.19.1 (official stack), enforce_eager, greedy", gate="McNemar p < 0.05 on test at W4A16, not lower at bf16")) as ex:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    from transformers import AutoTokenizer
    src = official_dir()
    tok = AutoTokenizer.from_pretrained(src)
    ex.log("official checkpoint", src, "files", sorted(os.listdir(src)))
    dev = bench.mbppplus("dev")
    test = bench.mbppplus("test")
    blocks = BLOCKS
    if SMOKE:
        dev, test, blocks = dev[:6], test[:6], BLOCKS[:1]
    ex.log("dev", len(dev), "test", len(test), "blocks", len(blocks))
    res = {}
    # pipeline check: the official folder and an identity rebuild must agree
    res["official"], dev_off = evaluate(src, "official_dev", dev, tok, ex, SCAN_MAX)
    t0 = time.time()
    if not os.path.exists(os.path.join(OUT, "rows_identity_dev.json")):
        rep = build(src, os.path.join(WORK, "m"), order_for(None, 0), False)
        ex.log("identity rebuild", rep, f"{time.time() - t0:.0f} s")
    res["identity"], _ = evaluate(os.path.join(WORK, "m"), "identity_dev", dev, tok, ex, SCAN_MAX)
    if abs(res["identity"]["passed"] - res["official"]["passed"]) > 3:
        raise RuntimeError(f"identity rebuild {res['identity']['passed']} vs official {res['official']['passed']}: the build changes the model")
    dev_pass = {}
    for s, k in blocks:
        tag = f"dup_s{s}_k{k}"
        t0 = time.time()
        try:
            if not os.path.exists(os.path.join(OUT, f"rows_{tag}_dev.json")):
                rep = build(src, os.path.join(WORK, "m"), order_for(s, k), s >= 15)
                ex.log(tag, "built", f"{time.time() - t0:.0f} s", "stacked", rep["stacked"])
            res[tag], dev_pass[tag] = evaluate(os.path.join(WORK, "m"), tag + "_dev", dev, tok, ex, SCAN_MAX)
        except Exception as e:  # noqa: BLE001  a block that fails to load is recorded, the scan goes on
            ex.log(tag, "FAILED", repr(e)[:1500])
            res[tag] = dict(error=repr(e)[:1500])
        json.dump(res, open(os.path.join(OUT, "scan.json"), "w"), indent=1)
    ok = {t: r for t, r in res.items() if t.startswith("dup_") and "passed" in r}
    if not ok:
        raise RuntimeError("no block loaded")
    best = max(ok, key=lambda t: (ok[t]["passed"], -int(t.split("_k")[1]), -int(t.split("_s")[1].split("_")[0])))
    bs, bk = int(best.split("_s")[1].split("_")[0]), int(best.split("_k")[1])
    same_k = [t for t in ok if t != best and t.endswith(f"_k{bk}")]
    control = random.Random(86).choice(sorted(same_k)) if same_k else None
    ex.log("selected", best, ok[best]["passed"], "dev; official", res["official"]["passed"], "| control", control)
    # confirmation on the test half
    conf = {}
    conf["official"], p_off = evaluate(src, "official_test", test, tok, ex)
    build(src, os.path.join(WORK, "m"), order_for(bs, bk), bs >= 15)
    conf[best], p_best = evaluate(os.path.join(WORK, "m"), best + "_test", test, tok, ex)
    if control:
        cs, ck = int(control.split("_s")[1].split("_")[0]), int(control.split("_k")[1])
        build(src, os.path.join(WORK, "m"), order_for(cs, ck), cs >= 15)
        conf[control], p_ctl = evaluate(os.path.join(WORK, "m"), control + "_test", test, tok, ex)
    ids = [t["task_id"] for t in test]
    x, y, p = mcnemar([p_best[i] for i in ids], [p_off[i] for i in ids])
    stats = dict(best_only=x, official_only=y, p=round(p, 4))
    if control:
        cx, cy, cp = mcnemar([p_best[i] for i in ids], [p_ctl[i] for i in ids])
        stats.update(best_vs_control=dict(best_only=cx, control_only=cy, p=round(cp, 4)))
    ex.log("test half W4A16", {t: r["passed"] for t, r in conf.items()}, stats)
    # bf16 precision check: the public QAT unquantized checkpoint, same build
    bf = {}
    try:
        from huggingface_hub import snapshot_download
        hf = snapshot_download("google/gemma-4-E2B-it-qat-q4_0-unquantized", local_dir="/content/bf16_src")
        bf["official_bf16"], b_off = evaluate(hf, "bf16_official_test", test, tok, ex)
        build(hf, os.path.join(WORK, "m"), order_for(bs, bk), bs >= 15)
        bf[best + "_bf16"], b_best = evaluate(os.path.join(WORK, "m"), best + "_bf16_test", test, tok, ex)
        bx, by, bp = mcnemar([b_best[i] for i in ids], [b_off[i] for i in ids])
        stats["bf16"] = dict(best_only=bx, official_only=by, p=round(bp, 4))
    except Exception as e:  # noqa: BLE001
        ex.log("bf16 check FAILED", repr(e)[:1500])
        stats["bf16"] = dict(error=repr(e)[:500])
    gate = stats["p"] < 0.05 and x > y and (stats["bf16"].get("best_only", 0) >= stats["bf16"].get("official_only", 1))
    ex.result(scan={t: (r.get("passed"), r.get("categories")) if "passed" in r else r for t, r in res.items()}, selected=best, control=control,
              test_w4a16={t: r["passed"] for t, r in conf.items()}, test_bf16={t: r["passed"] for t, r in bf.items()}, stats=stats, gate_passed=gate,
              headline=f"selected {best}: dev {ok[best]['passed']} vs official {res['official']['passed']} of {len(dev)}; test W4A16 "
                       f"{conf[best]['passed']} vs {conf['official']['passed']} of {len(test)} (McNemar p = {stats['p']}); gate {'passed' if gate else 'not passed'}")

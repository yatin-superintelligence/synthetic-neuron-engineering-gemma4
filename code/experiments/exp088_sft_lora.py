"""EXP-088 (training track, NOT training-free): supervised fine-tuning of a LoRA for E2B on verified bug-fix tasks.

The author asked (2026-11-08) for a trained comparison next to the training-free work: build a training set of at least 500 tasks and
train E2B with SFT (and later GRPO). This is that SFT step. It is reported separately from the training-free results.

Training set: tools/trainset/kaggle_build.py (Kaggle CPU, official verify_task): small bugs injected into the newest non-evaluation
snapshots of fastapi, rich and requests, each kept only if it makes 1 to 25 tests fail and its reference fix resolves under the
official verification; no injected bug lies inside a function or class named in the 58 evaluation fixes.

Each task becomes one scripted trajectory in the official agent's own format: the official system prompt and the task prompt built by
swegemma's build_agent_prompt (with the workspace tree from its own command), the official tool declarations (sne.loops.TOOLS) and
chat template with thinking kept, and five tool calls whose results are real where they exist: run_command on the failing tests (the
failing output recorded by the official verification), run_command grep for the function (run here on the buggy snapshot), read_file
around the bug (the buggy file itself), edit_file restoring the original lines (diff as the tool reports it), run_command on the
failing tests again (passing), submit_patch. The loss is on the assistant turns only.

LoRA: rank 16, alpha 32, on gate_proj, up_proj, down_proj and o_proj of every language-model layer; 2 epochs, AdamW lr 1e-4, cosine,
20 warmup steps, effective batch 8; 5% of tasks held out for a validation loss. Base: the public QAT unquantized E2B in bf16 (the
same reference model as every closed-form edit); the adapter is then loaded by the official vLLM on the official W4A16 checkpoint as
a check before any agent run.
Pre-registered evaluation (D-060): official E2B W4A16 with this adapter at seeds 1 to 3 on the Colab L4, scored on Kaggle, against
unedited E2B at the same seeds on the L4; a gain is claimed if pooled tasks solved rise with a task-level sign-flip p < 0.05.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sne import ROOT, SMOKE  # noqa: E402
from sne.loops import TOOLS  # noqa: E402
from sne.registry import Experiment  # noqa: E402

COMP = "/content/kaggle_input/competitions/gemma-4-developer-agent"
WORK = "/content/sft"
OUT = os.path.join(ROOT, "artifacts_local", "EXP-088")
QAT = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
PKG = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "src/requests"}
DECLARED = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch", "get_code_neighbors", "search_similar_code",
            "get_code_subgraph", "code_analyzer_agent"]
R, ALPHA, LR, EPOCHS, ACC, WARM, MAXLEN = 16, 32, 1e-4, 2, 8, 20, 12288
TARGET = r".*language_model\.layers\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.o_proj)$"
WHY = dict(compare="the comparison operator is wrong", boolop="the two conditions are combined with the wrong boolean operator",
           **{"not": "the condition is negated when it should not be"}, offbyone="an index or bound is off by one",
           arith="the arithmetic uses the wrong operator", guard="an early-exit guard is missing",
           args="the arguments are passed in the wrong order", boolconst="a boolean flag has the wrong value")


def obs(d):
    return json.dumps(d)


def _lora_class():
    import torch

    class LoRALinear(torch.nn.Module):
        """y = W x + (alpha / r) * B A x, with A random (Kaiming) and B zero at the start, as PEFT initialises a LoRA."""

        def __init__(self, base, r, alpha, dropout=0.05):
            super().__init__()
            self.base = base
            self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
            self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scale = alpha / r
            self.drop = torch.nn.Dropout(dropout)

        def forward(self, x):
            y = self.base(x)
            z = (self.drop(x).to(torch.float32) @ self.A.t()) @ self.B.t()
            return y + (z * self.scale).to(y.dtype)

    return LoRALinear


def LoRALinear(base, r, alpha):  # noqa: N802
    return _lora_class()(base, r, alpha)


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)


def materialize(iid, repo):
    from swegemma.evaluate import resolve_task_snapshot_paths
    snap, bsnap, ppath = resolve_task_snapshot_paths(os.path.join(COMP, "snapshots"), iid, repo)
    d = os.path.join(WORK, "src_" + PKG[repo].split("/")[-1])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    subprocess.run(["tar", "--no-same-owner", "-xzf", str(bsnap or snap), "-C", d], check=True)
    if ppath:
        r = run(f"git apply --binary --whitespace=nowarn {ppath}", d)
        if r.returncode != 0:
            subprocess.run(["patch", "-p1", "-i", str(ppath)], cwd=d, check=True)
    run("git add -A && git -c user.email=a@b -c user.name=sne commit -qm base --allow-empty", d)
    return d


def trajectory(task, d, tree, system, cfg):
    """The messages of one scripted trajectory on the buggy snapshot in d (the bug patch is applied by the caller)."""
    from swegemma.harness.agent_runner import build_agent_prompt
    from swegemma.models.task import Task
    t = Task(instance_id=task["instance_id"], repo=task["repo"], base_commit=task["base_commit"], problem_statement=task["problem_statement"])
    user = build_agent_prompt(t, cfg, tree, declared_tools=DECLARED)
    f2p, files, fpath = task["FAIL_TO_PASS"], task["test_files"], task["file"]
    name = task["function"].split(".")[-1]
    exc = task["buggy_excerpt"]
    fix = task["patch"]
    diff = "\n".join(l for l in fix.splitlines() if not l.startswith("diff --git")) + "\n"
    before, after = (task.get("before") or "").rstrip("\n"), (task.get("after") or "").rstrip("\n")
    if task["operator"] == "guard" or not after.strip():
        return None
    # old_string must occur exactly once in the file, as the real edit_file requires: widen by context lines until it does
    NL = chr(10)
    text = open(os.path.join(d, fpath), encoding="utf-8").read()
    buggy = text.split(NL)
    L, a_lines, b_lines = task["lineno"] - 1, after.split(NL), before.split(NL)
    if buggy[L:L + len(a_lines)] != a_lines:
        return None
    lo, hi = L, L + len(a_lines)
    old = NL.join(buggy[lo:hi])
    while text.count(old) != 1 and (lo > 0 or hi < len(buggy)):
        lo, hi = max(0, lo - 1), min(len(buggy), hi + 1)
        old = NL.join(buggy[lo:hi])
    if text.count(old) != 1:
        return None
    new = NL.join(buggy[lo:L] + b_lines + buggy[L + len(a_lines):hi])
    t1 = f"python -m pytest {' '.join(files)} -q"
    fail = task["fail_output"][:5000]
    g = run(f"grep -rn 'def {name}' {PKG[task['repo']]}", d)
    grep_out = g.stdout[:5000] or f"{fpath}: (no match)"
    t2 = f"python -m pytest {' '.join(f2p[:6])} -q"
    n = min(len(f2p), 6)
    passed = "." * n + " " * max(1, 70 - n) + "[100%]\n" + f"{n} passed in 0.{random.Random(task['instance_id']).randint(21, 89)}s\n"
    why = WHY.get(task["operator"], "the logic is wrong")
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "reasoning": "The issue names failing tests. I will run them first to see the exact failure.",
         "tool_calls": [{"type": "function", "function": {"name": "run_command", "arguments": {"command": t1}}}]},
        {"role": "tool", "name": "run_command", "content": obs(dict(status="error", error_type="CommandError", error_message="",
                                                                  details=dict(stdout=fail, stderr="", exit_code=1)))},
        {"role": "assistant", "content": "", "reasoning": f"The failure goes through `{name}`. I will find where it is defined in the library.",
         "tool_calls": [{"type": "function", "function": {"name": "run_command", "arguments": {"command": f"grep -rn 'def {name}' {PKG[task['repo']]}"}}}]},
        {"role": "tool", "name": "run_command", "content": obs(dict(status="ok", stdout=grep_out, stderr="", exit_code=0))},
        {"role": "assistant", "content": "", "reasoning": f"`{name}` is in `{fpath}`. I will read the code around it.",
         "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"filepath": fpath, "start_line": exc["start_line"],
                                                                                              "end_line": exc["end_line"]}}}]},
        {"role": "tool", "name": "read_file", "content": obs(dict(status="ok", filepath=fpath, content=exc["content"], start_line=exc["start_line"],
                                                                end_line=exc["end_line"], total_lines=exc["total_lines"],
                                                                is_truncated=exc["end_line"] < exc["total_lines"]))},
        {"role": "assistant", "content": "", "reasoning": f"In `{task['function']}`, {why}: `{old.strip()}` should be `{new.strip()}`. "
                                                           "This matches the failing assertion. I will make this one-line fix.",
         "tool_calls": [{"type": "function", "function": {"name": "edit_file", "arguments": {"filepath": fpath, "old_string": old, "new_string": new}}}]},
        {"role": "tool", "name": "edit_file", "content": obs(dict(status="ok", filepath=fpath, occurrences=1, strategy="exact", diff=diff,
                                                                is_truncated=False))},
        {"role": "assistant", "content": "", "reasoning": "Now I will rerun the failing tests to confirm the fix.",
         "tool_calls": [{"type": "function", "function": {"name": "run_command", "arguments": {"command": t2}}}]},
        {"role": "tool", "name": "run_command", "content": obs(dict(status="ok", stdout=passed, stderr="", exit_code=0))},
        {"role": "assistant", "content": "", "reasoning": "The failing tests pass now. The fix is minimal, so I will submit it.",
         "tool_calls": [{"type": "function", "function": {"name": "submit_patch", "arguments": {}}}]},
    ]
    return msgs


def render(tok, msgs, gen):
    s = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=gen, enable_thinking=True, preserve_thinking=True)
    return s


def encode(tok, msgs):
    """Token ids and labels: loss on each assistant turn only (prefix-consistent rendering is checked)."""
    full = render(tok, msgs, False)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        a_txt = render(tok, msgs[:i], True)
        b_txt = render(tok, msgs[:i + 1], False)
        if not full.startswith(a_txt) or not full.startswith(b_txt):
            return None, None
        a = len(tok(a_txt, add_special_tokens=False)["input_ids"])
        b = len(tok(b_txt, add_special_tokens=False)["input_ids"])
        for j in range(a, min(b, len(ids))):
            labels[j] = ids[j]
    return ids, labels


def build_data(tasks, ex):
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.harness.agent_runner import build_workspace_tree_command
    cfg = EvalConfig(tasks_path=os.path.join(COMP, "tasks.jsonl"), snapshots_dir=os.path.join(COMP, "snapshots"), results_dir="/content/sft/v",
                     submission_dir="/content/sft", models={}, sandbox="subprocess", timeout_seconds=300, verification_timeout_seconds=300,
                     max_time_minutes=60, max_tool_calls=100, max_turns=None, limits=build_submission_limits()[0],
                     generation_constraints=build_submission_limits()[1], verbose=False, concurrency=1)
    sysp = next((p for p in glob.glob(os.path.join(COMP, "**", "prompts", "system.md"), recursive=True)), None) \
        or os.path.join(ROOT, "data", "kaggle", "e2b_submission", "prompts", "system.md")
    system = open(sysp, encoding="utf-8").read()
    ex.log("system prompt", sysp, len(system), "chars")
    out, skipped = [], {}
    by_repo = {}
    for t in tasks:
        by_repo.setdefault((t["repo"], t["base_instance"]), []).append(t)
    for (repo, base), ts in by_repo.items():
        d = materialize(base, repo)
        tree = run(build_workspace_tree_command(workspace_dir=d), d).stdout
        for t in ts:
            bp = os.path.join(WORK, "bug.patch")
            open(bp, "w", encoding="utf-8").write(t["bug_patch"])
            if run(f"git apply {bp}", d).returncode != 0:
                skipped["bug patch does not apply"] = skipped.get("bug patch does not apply", 0) + 1
                continue
            try:
                m = trajectory(t, d, tree, system, cfg)
            finally:
                run(f"git apply -R {bp}", d)
            if m is None:
                skipped["no single-line edit"] = skipped.get("no single-line edit", 0) + 1
                continue
            out.append(dict(instance_id=t["instance_id"], messages=m))
        ex.log(repo, "trajectories so far", len(out), "skipped", skipped)
    return out, skipped


def train(examples, ex):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from huggingface_hub import snapshot_download
    src = snapshot_download(QAT, local_dir="/content/bf16_src")  # the folder EXP-086 filled, when chained on its runtime
    tok = AutoTokenizer.from_pretrained(src)
    enc, bad, long_ = [], 0, 0
    for e in examples:
        ids, lab = encode(tok, e["messages"])
        if ids is None:
            bad += 1
            continue
        if len(ids) > MAXLEN:
            long_ += 1
            continue
        enc.append(dict(iid=e["instance_id"], ids=ids, labels=lab))
    ex.log("encoded", len(enc), "| render mismatch", bad, "| over", MAXLEN, "tokens", long_,
           "| median tokens", sorted(len(x["ids"]) for x in enc)[len(enc) // 2] if enc else None,
           "| median target tokens", sorted(sum(l != -100 for l in x["labels"]) for x in enc)[len(enc) // 2] if enc else None)
    rng = random.Random(88)
    rng.shuffle(enc)
    n_val = max(1, len(enc) // 20)
    val, tr = enc[:n_val], enc[n_val:]
    json.dump(dict(train=[x["iid"] for x in tr], val=[x["iid"] for x in val]), open(os.path.join(OUT, "split.json"), "w"))
    model = Gemma4ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16)
    # added for EXP-110b (2026-11-10, D-152): long conversations ran out of the L4's 22 GB at 13,865 tokens. The per-layer embedding
    # table (about 4.7 GB, frozen, lookup only) stays in CPU RAM and only its lookups go to the GPU (sne.model's ple_on_cpu), and the
    # unused vision and audio encoders are dropped; the forward pass is numerically unchanged.
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "src"))
    from sne.model import _CPUEmbedding
    from sne.chunked_attention import install as _chunked_sdpa
    _chunked_sdpa()  # head_dim-512 full-attention layers: query-chunked SDPA (exact), see src/sne/chunked_attention.py
    for _nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if getattr(model.model, _nm, None) is not None:
            setattr(model.model, _nm, None)
    _tm = model.model.language_model
    _ple = getattr(_tm, "embed_tokens_per_layer", None)
    if _ple is not None:
        _tm.embed_tokens_per_layer = torch.nn.Identity()
    model = model.cuda()
    if _ple is not None:
        _tm.embed_tokens_per_layer = _CPUEmbedding(_ple, "cuda")
    ex.log("GPU memory after load", round(torch.cuda.memory_allocated() / 1e9, 2), "GB; text attention", model.config.get_text_config()._attn_implementation)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    loras = {}
    for name, mod in list(model.named_modules()):
        if re.fullmatch(TARGET, name) and isinstance(mod, torch.nn.Linear):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            w = LoRALinear(mod, R, ALPHA)
            setattr(parent, name.rsplit(".", 1)[1], w)
            loras[name] = w
    n_train = sum(p.numel() for w in loras.values() for p in (w.A, w.B))
    ex.log("LoRA modules", len(loras), "trainable parameters", n_train)
    pm = model
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.enable_input_require_grads()
    lm = model
    head, cap = lm.lm_head, lm.config.get_text_config().final_logit_softcapping
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    steps = math.ceil(len(tr) * EPOCHS / ACC)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    import torch.utils.checkpoint

    def loss_of(x):
        ids = torch.tensor([x["ids"]], device="cuda")
        lab = torch.tensor([x["labels"]], device="cuda")
        h = lm.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
        tgt = lab[0, 1:]
        keep = tgt != -100
        h, tgt = h[keep], tgt[keep]
        tot = torch.zeros((), device="cuda")

        def chunk_loss(hc, tc):
            lg = head(hc).float()
            if cap:
                lg = torch.tanh(lg / cap) * cap
            return torch.nn.functional.cross_entropy(lg, tc, reduction="sum")

        for k in range(0, h.shape[0], 512):
            # each chunk's 262,144-way logits are recomputed in the backward pass instead of stored: stored, they cost about 1.5 MB
            # per trained token and ran the L4 out of memory past about 12k tokens (EXP-110b, D-152c); the loss is unchanged
            hc, tc = h[k:k + 512], tgt[k:k + 512]
            tot = tot + (torch.utils.checkpoint.checkpoint(chunk_loss, hc, tc, use_reentrant=False) if torch.is_grad_enabled() else chunk_loss(hc, tc))
        return tot / max(1, int(keep.sum())), int(keep.sum())

    def evaluate_val():
        pm.eval()
        with torch.no_grad():
            ls = [loss_of(x)[0].item() for x in val]
        pm.train()
        return round(sum(ls) / len(ls), 4)

    hist = dict(val_before=evaluate_val(), train=[], val=[])
    ex.log("validation loss before training", hist["val_before"], "| train", len(tr), "val", len(val), "| optimizer steps", steps)
    pm.train()
    step, t0 = 0, time.time()
    order = [x for _ in range(EPOCHS) for x in rng.sample(tr, len(tr))]
    if SMOKE:
        order = order[:ACC * 2]
    run_loss = []
    oom = 0
    for i, x in enumerate(order):
        failed = False
        try:
            l, _ = loss_of(x)
            (l / ACC).backward()
            run_loss.append(l.item())
        except torch.cuda.OutOfMemoryError:  # added for EXP-110b (long conversations): skip the example, keep the run
            failed = True
        if failed:  # outside the handler, so the exception's frames (holding the failed step's tensors) are already released
            import gc as _gc
            l = None
            opt.zero_grad(set_to_none=True)  # drop the failed example's partial gradients (and this accumulation group's)
            _gc.collect()
            torch.cuda.empty_cache()
            oom += 1
            ex.log(f"[sft] out of memory on an example of {len(x['ids'])} tokens; skipped ({oom} so far)")
        if (i + 1) % ACC == 0:
            torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 and run_loss:
                hist["train"].append((step, round(sum(run_loss) / len(run_loss), 4)))
                ex.log(f"[sft] step {step}/{steps} train loss {hist['train'][-1][1]} | {(time.time() - t0) / 60:.1f} min")
                run_loss = []
            if step % 50 == 0 or step == steps:
                hist["val"].append((step, evaluate_val()))
                ex.log(f"[sft] step {step} validation loss {hist['val'][-1][1]}")
    dst = os.path.join(OUT, "sft_lora")
    from safetensors.torch import save_file
    os.makedirs(dst, exist_ok=True)
    sd = {}
    for name, w in loras.items():  # the PEFT key layout our closed-form adapters already load in vLLM
        sd[f"base_model.model.{name}.lora_A.weight"] = w.A.detach().float().cpu().contiguous()
        sd[f"base_model.model.{name}.lora_B.weight"] = w.B.detach().float().cpu().contiguous()
    save_file(sd, os.path.join(dst, "adapter_model.safetensors"))
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path="google/gemma-4-E2B-it", r=R, lora_alpha=ALPHA,
               target_modules=sorted({n.rsplit(".", 1)[1] for n in loras}), lora_dropout=0.0, bias="none", fan_in_fan_out=False,
               init_lora_weights=True, inference_mode=True, use_rslora=False)
    json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"), indent=2)
    return hist, dst, len(tr), len(val)


def vllm_check(dst, ex):
    """The adapter must load in the official vLLM on the official W4A16 checkpoint and change nothing else."""
    md = next(p for p in glob.glob("/content/kaggle_input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/*") if os.path.exists(os.path.join(p, "config.json")))
    log = os.path.join(OUT, "vllm_lora_check.log")
    cmd = ["/opt/sne/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--model", md, "--served-model-name", "base", "--port", "8012",
           "--max-model-len", "8192", "--enforce-eager", "--enable-lora", "--max-lora-rank", str(R), "--lora-modules", f"sft={dst}",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0}), "--gpu-memory-utilization", "0.85"]
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    import urllib.request
    try:
        for _ in range(240):
            if p.poll() is not None:
                raise RuntimeError("vLLM exited: " + open(log, errors="replace").read()[-2000:])
            try:
                urllib.request.urlopen("http://127.0.0.1:8012/health", timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(5)
        out = {}
        for m in ("base", "sft"):
            body = json.dumps(dict(model=m, prompt="def add(a, b):\n    return", max_tokens=8, temperature=0)).encode()
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8012/v1/completions", body, {"Content-Type": "application/json"}), timeout=300))
            out[m] = r["choices"][0]["text"]
        ex.log("vLLM LoRA check", out)
        return out
    finally:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()


with Experiment("EXP-088", "sft-lora-trainset", category="training", depends_on=["EXP-044a"],
                motivation="Training track requested by the author: an SFT LoRA on verified bug-fix tasks, as a trained comparison for the training-free edits.",
                hypothesis="SFT on scripted trajectories over at least 500 verified injected-bug tasks raises E2B's official tasks solved at seeds 1 to 3 "
                           "against unedited E2B at the same seeds (task-level sign-flip p < 0.05).",
                config=dict(rank=R, alpha=ALPHA, lr=LR, epochs=EPOCHS, accumulation=ACC, warmup=WARM, max_len=MAXLEN, target=TARGET, base=QAT)) as ex:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "artifacts_local", "trainset", "*", "trainset", "tasks.jsonl")))
    tasks = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
    tasks = [t for t in tasks if t.get("source") == "synthetic" and isinstance(t.get("buggy_excerpt"), dict) and "content" in t["buggy_excerpt"]]
    # the two builds of 2026-11-08 overlapped (both ran every repository): one task per distinct injected bug, with ids made unique
    import hashlib
    uniq = {}
    for t in tasks:
        key = hashlib.sha1(t["bug_patch"].encode("utf-8")).hexdigest()[:8]
        if key not in uniq:
            t["instance_id"] = f"{t['instance_id']}_{key}"
            uniq[key] = t
    n_all, tasks = len(tasks), list(uniq.values())
    ex.log("training tasks", len(tasks), "distinct of", n_all, "from", files, "| per repo",
           {r: sum(t["repo"] == r for t in tasks) for r in sorted({t["repo"] for t in tasks})})
    if SMOKE:
        tasks = tasks[:12]
    examples, skipped = build_data(tasks, ex)
    with open(os.path.join(OUT, "sft_examples.jsonl"), "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    hist, dst, n_tr, n_val = train(examples, ex)
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    check = vllm_check(dst, ex)
    ex.result(tasks=len(tasks), trajectories=len(examples), skipped=skipped, train=n_tr, val=n_val, history=hist, vllm_check=check, adapter=dst,
              headline=f"SFT LoRA r{R} on {n_tr} trajectories: validation loss {hist['val_before']} -> {hist['val'][-1][1] if hist['val'] else None}")

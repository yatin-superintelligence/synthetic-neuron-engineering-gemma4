"""EditBench: replay of the agent state before each edit_file call in official traces, scored with the official matcher.

Traces hold competition content, so contexts stay under the git-ignored data/kaggle/ folder; committed outputs hold ids
and numbers only. Each trace is walked with a virtual file system, the task snapshot plus every edit that succeeded, so
a replayed old_string is tested against the file as it stood at that step. Shell commands that change files are not
tracked; run_command edits are rare in the traces and are counted by `walk`.
"""
import collections
import glob
import importlib.util
import json
import os
import random
import re
import sys
import tarfile
import types
from concurrent.futures import ThreadPoolExecutor

from . import DATA, ROOT, SMOKE
from .loops import SLOT_CTX, STOP, THINK_BUDGET, agent_steps, generate, loop_stats, messages_before, render, split_map
from .stats import mcnemar_exact
from .sweagent import CALL, parse_args

EDITBENCH = os.path.join(DATA, "kaggle", "editbench_v1.jsonl")
SNAPSHOTS = os.path.join(DATA, "kaggle", "snapshots")


def _load_official():
    """Loads the official matcher file by file, since the package __init__ pulls in the whole harness."""
    base = os.path.join(ROOT, "tools", "harness", "src_adk_eval_core", "adk_eval_core")
    pkg = types.ModuleType("adk_eval_core")
    pkg.__path__ = [base]
    sys.modules.setdefault("adk_eval_core", pkg)
    for name, rel in [("adk_eval_core.errors", "errors.py"), ("official_edit", os.path.join("editing", "edit.py"))]:
        spec = importlib.util.spec_from_file_location(name, os.path.join(base, rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


official = _load_official()


def applies(content, old, new):
    """Outcome of the official matcher: 'ok:<strategy>', 'not_found', 'identical', 'multiple', or 'empty'."""
    if not old:
        return "empty"
    r = official.apply_replacement(content, old, new, allow_multiple=False)
    if not r.error_message:
        return f"ok:{r.strategy}"
    m = r.error_message
    if r.occurrences > 1:
        return "multiple"
    return "identical" if "No changes" in m or "identical" in m else "not_found"


def unescape(s):
    return s.replace('\\\\\n', "\n").replace('\\\n', "\n").replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t")


class VirtualFS:
    def __init__(self, iid):
        self.tar = tarfile.open(os.path.join(SNAPSHOTS, f"{iid}.tgz"))
        self.names = {m.name.split("/", 1)[-1]: m for m in self.tar.getmembers() if m.isfile()}
        self.files = {}

    def read(self, fp):
        fp = (fp or "").removeprefix("/workspace/").lstrip("./")
        if fp in self.files:
            return self.files[fp]
        m = self.names.get(fp)
        return self.tar.extractfile(m).read().decode("utf-8", "replace") if m else None

    def write(self, fp, text):
        self.files[(fp or "").removeprefix("/workspace/").lstrip("./")] = text


def obs(step):
    try:
        c = json.loads((step.get("observation") or {}).get("content") or "{}")
        return c if isinstance(c, dict) else {}
    except (TypeError, ValueError):
        return {}


def walk(trace, iid):
    """Yields (index, step, args, content before the edit, original status) for every edit_file call, tracking the file state."""
    fs = VirtualFS(iid)
    for i, s in agent_steps(trace):
        for c in s.get("tool_calls") or []:
            name, a = c.get("function_name"), c.get("arguments") or {}
            res = obs(s)
            if name == "edit_file":
                before = fs.read(a.get("filepath"))
                yield i, s, a, before, res
                if res.get("status") != "error" and before is not None:
                    try:
                        fs.write(a.get("filepath"), official.apply_replacement(before, a.get("old_string") or "", a.get("new_string") or "",
                                                                                 allow_multiple=bool(a.get("allow_multiple"))).new_content)
                    except Exception:
                        pass
            elif name == "write_file" and res.get("status") != "error":
                fs.write(a.get("filepath"), a.get("content") or "")


def classify(old, new, content, status, snapshot):
    """Reason a recorded edit failed, in priority order, using the official matcher."""
    if status != "error":
        return "applied"
    if content is None:
        return "file_missing"
    r = applies(content, old, new)
    if r.startswith("ok"):
        return "applies_now"  # the walk disagrees with the record, for example after an untracked shell edit
    if r in ("identical", "multiple", "empty"):
        return r
    if old != unescape(old) and applies(content, unescape(old), unescape(new)).startswith(("ok", "multiple", "identical")):
        return "over_escaped"
    if applies(snapshot or "", old, new).startswith(("ok", "multiple")):
        return "stale_after_own_edit"
    ol = [x.strip() for x in unescape(old).splitlines() if x.strip()]
    lines = {x.strip() for x in content.splitlines()}
    frac = sum(x in lines for x in ol) / max(1, len(ol))
    if frac == 1:
        return "lines_real_block_not"  # every line exists but not contiguously or with other indentation
    return "mostly_real_altered" if frac >= 0.5 else "invented"


def census(trace_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(trace_dir, "trace_*.json"))):
        iid = os.path.basename(f)[6:-5]
        t = json.load(open(f, encoding="utf-8"))
        prev = None
        snap = VirtualFS(iid)
        for i, s, a, before, res in walk(t, iid):
            old, new = a.get("old_string") or "", a.get("new_string") or ""
            kind = classify(old, new, before, res.get("status"), snap.read(a.get("filepath")))
            key = (a.get("filepath"), old, new)
            rows.append(dict(instance_id=iid, step=i, kind=kind, repeat_of_previous=key == prev, old_chars=len(old), has_backslash="\\" in old))
            prev = key
    return rows


def summarise(rows):
    c = collections.Counter(r["kind"] for r in rows)
    fail = [r for r in rows if r["kind"] != "applied"]
    return dict(edits=len(rows), applied=c["applied"], failed=len(fail), kinds=dict(c.most_common()),
                failed_repeating_identical_call=sum(r["repeat_of_previous"] for r in fail), failed_with_backslash=sum(r["has_backslash"] for r in fail))


def build(trace_dirs):
    """One case per recorded edit_file call, with the full context and the file overlay at that step."""
    split = split_map()
    out = []
    for run, d in trace_dirs.items():
        for f in sorted(glob.glob(os.path.join(d, "trace_*.json"))):
            iid = os.path.basename(f)[6:-5]
            t = json.load(open(f, encoding="utf-8"))
            snap, ops = VirtualFS(iid), {}
            for i, s, a, before, res in walk(t, iid):
                old, new = a.get("old_string") or "", a.get("new_string") or ""
                kind = classify(old, new, before, res.get("status"), snap.read(a.get("filepath")))
                if kind != "file_missing" and i >= 2:
                    first = next((k for k in a if k in ("old_string", "new_string")), None)
                    out.append(dict(case=f"{run}/{iid}/{i}", run=run, instance_id=iid, step=i, kind=kind, ok=kind == "applied", split=split[iid],
                                    filepath=a.get("filepath"), first_key=first, thought=s.get("message") or "", overlay=dict(ops), messages=messages_before(t, i),
                                    args={k: v for k, v in a.items() if isinstance(v, str)}))
                if kind == "applied" and before is not None:
                    ops[(a.get("filepath") or "").removeprefix("/workspace/").lstrip("./")] = official.apply_replacement(
                        before, old, new, allow_multiple=bool(a.get("allow_multiple"))).new_content
    with open(EDITBENCH, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def load(split=None, per_run_task=None, seed=20261106):
    """Cases of a split. per_run_task keeps a seeded sample of states from each task in each run, because a task that retries
    one edit many times would otherwise dominate the sample."""
    rows = [json.loads(l) for l in open(EDITBENCH, encoding="utf-8")]
    rows = [r for r in rows if split is None or r["split"] == split]
    if per_run_task:
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["run"], r["instance_id"])].append(r)
        rng = random.Random(seed)
        return [r for k in sorted(by) for r in rng.sample(by[k], min(per_run_task, len(by[k])))]
    return rows[:6] if SMOKE else rows


def score(text, case):
    """Outcome of a replayed response: its first tool call, tested against the file as it stood at that step."""
    m = CALL.search(text)
    if not m:
        return dict(outcome="no_tool", tool=None, backslash=False)
    args = parse_args(m.group(2)) or {}
    if m.group(1) != "edit_file":
        return dict(outcome="other_tool", tool=m.group(1), backslash=False)
    fs = VirtualFS(case["instance_id"])
    fs.files = dict(case["overlay"])
    content = fs.read(args.get("filepath"))
    old, new = str(args.get("old_string") or ""), str(args.get("new_string") or "")
    if content is None:
        return dict(outcome="file_missing", tool="edit_file", backslash=chr(92) in old)
    r = applies(content, old, new)
    return dict(outcome="applied" if r.startswith("ok") else r, strategy=r.split(":")[-1], tool="edit_file", backslash=chr(92) in old,
                same_file=args.get("filepath") == case["filepath"])


def build_retries(trace_dirs):
    """One case per failed edit_file call: the state right after the failure, when the agent decides what to do next."""
    split = split_map()
    out = []
    for run, d in trace_dirs.items():
        for f in sorted(glob.glob(os.path.join(d, "trace_*.json"))):
            iid = os.path.basename(f)[6:-5]
            t = json.load(open(f, encoding="utf-8"))
            ops = {}
            snap = VirtualFS(iid)
            for i, s, a, before, res in walk(t, iid):
                old, new = a.get("old_string") or "", a.get("new_string") or ""
                kind = classify(old, new, before, res.get("status"), snap.read(a.get("filepath")))
                if kind == "applied" and before is not None:
                    ops[(a.get("filepath") or "").removeprefix("/workspace/").lstrip("./")] = official.apply_replacement(
                        before, old, new, allow_multiple=bool(a.get("allow_multiple"))).new_content
                elif kind not in ("applied", "file_missing"):
                    out.append(dict(case=f"{run}/{iid}/{i}", run=run, instance_id=iid, step=i, kind=kind, split=split[iid], filepath=a.get("filepath"),
                                    failed_old=old, overlay=dict(ops), messages=messages_before(t, i + 1)))
    return out


def retry_run(server, tok, cases, lora_scales=None, seeds=(0,), temperature=0.2, top_p=0.95, budget=THINK_BUDGET):
    """What the agent does after a failed edit: resend the same text, try new text, or use another tool."""
    if SMOKE:
        seeds, budget = seeds[:1], min(budget, 128)
    res, texts = [], {}
    for c in cases:
        p = render(tok, c)
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        if n + 1024 > SLOT_CTX:
            continue
        for sd in seeds:
            sampling = dict(temperature=temperature, top_p=top_p, top_k=0, seed=sd, stop=STOP, cache_prompt=True)
            r = generate(server, p, n, sampling, lora_scales, budget, early_exit=True)
            text = r.pop("text")
            m = CALL.search(text)
            args = (parse_args(m.group(2)) or {}) if m else {}
            same = bool(m) and m.group(1) == "edit_file" and str(args.get("old_string") or "") == c["failed_old"]
            res.append(dict(case=c["case"], original=c["kind"], seed=sd, n_prompt=n, n_gen=r["n_gen"], **score(text, c), same_text=same, loop=loop_stats(text)["loop"]))
            texts[f"{c['case']}#{sd}"] = text
    n = max(1, len(res))
    return res, texts, dict(n=len(res), same_text=sum(r["same_text"] for r in res), applied=sum(r["outcome"] == "applied" for r in res), other_tool=sum(r["outcome"] == "other_tool" for r in res),
                            no_tool=sum(r["outcome"] == "no_tool" for r in res), loops=sum(r["loop"] for r in res), mean_tokens=round(sum(r["n_gen"] for r in res) / n, 1))


def with_system_suffix(tok, case, suffix):
    msgs = [dict(m) for m in case["messages"]]
    msgs[0]["content"] = (msgs[0].get("content") or "") + suffix
    return render(tok, dict(case, messages=msgs))


KEY_POINT = re.compile(r'<\|tool_call>call:edit_file\{filepath:<\|"\|>.*?<\|"\|>,', re.S)


def short_cases(tok, cases, max_tokens):
    """Keeps states whose full rendered prompt fits max_tokens. Long prompts take minutes each on this GPU, so local screens
    use early states of a task. The full prompt is always sent, so nothing is cut from any state that is kept."""
    kept = [c for c in cases if len(tok(render(tok, c), add_special_tokens=False)["input_ids"]) <= max_tokens]
    return kept[:6] if SMOKE else kept


def order_branches(server, tok, cases, seeds=(0, 1), temperature=0.2, top_p=0.95, lora_scales=None):
    """Causal test of argument order. From the recorded thought and file path, one branch must write old_string first and
    the other new_string first. The prefix is identical up to that token and thinking is not regenerated, so order is the
    only difference, and the test costs a fraction of a full replay."""
    if SMOKE:
        seeds = seeds[:1]
    jobs = []
    for c in cases:
        p = render(tok, c) + c["thought"] + f'<channel|><|tool_call>call:edit_file{{filepath:<|"|>{c["filepath"]}<|"|>,'
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        if n + 2048 <= SLOT_CTX:
            jobs.append((c, p))

    def one(job):
        c, p = job
        rows = []
        for sd in seeds:
            row = dict(case=c["case"], instance_id=c["instance_id"], original=c["kind"], original_first=c["first_key"], seed=sd)
            for key, name in (("new_string", "new_first"), ("old_string", "old_first")):
                head = f'<|tool_call>call:edit_file{{filepath:<|"|>{c["filepath"]}<|"|>,{key}:<|"|>'
                sampling = dict(temperature=temperature, top_p=top_p, top_k=0, seed=sd, stop=STOP, cache_prompt=True)
                g = server.complete(p + f'{key}:<|"|>', 128 if SMOKE else 4096, lora_scales, sampling)
                r = score(head + g["text"], c)
                row[name], row[name + "_tokens"], row[name + "_backslash"] = r["outcome"], g["n_gen"], r["backslash"]
            rows.append(row)
        return rows
    with ThreadPoolExecutor(server.parallel) as ex:
        res = [r for rows in ex.map(one, jobs) for r in rows]
    return res, order_summary(res)


def free_key(server, tok, cases, lora_scales=None, seeds=(0, 1), temperature=0.2, top_p=0.95):
    """Evaluates an order edit. From the recorded thought and file path the model chooses its own next argument. Returns
    how often it writes old_string first and how often the resulting edit applies."""
    if SMOKE:
        seeds = seeds[:1]
    jobs = []
    for c in cases:
        p = render(tok, c) + c["thought"] + f'<channel|><|tool_call>call:edit_file{{filepath:<|"|>{c["filepath"]}<|"|>,'
        if len(tok(p, add_special_tokens=False)["input_ids"]) + 2048 <= SLOT_CTX:
            jobs.append((c, p))

    def one(job):
        c, p = job
        rows = []
        for sd in seeds:
            sampling = dict(temperature=temperature, top_p=top_p, top_k=0, seed=sd, stop=STOP, cache_prompt=True)
            g = server.complete(p, 128 if SMOKE else 4096, lora_scales, sampling)
            head = f'<|tool_call>call:edit_file{{filepath:<|"|>{c["filepath"]}<|"|>,'
            r = score(head + g["text"], c)
            rows.append(dict(case=c["case"], instance_id=c["instance_id"], original=c["kind"], seed=sd, first_key=g["text"][:10], outcome=r["outcome"],
                             backslash=r["backslash"], tokens=g["n_gen"]))
        return rows
    with ThreadPoolExecutor(server.parallel) as ex:
        res = [r for rows in ex.map(one, jobs) for r in rows]
    n = max(1, len(res))
    return res, dict(n=len(res), old_first=round(sum(r["first_key"].startswith("old") for r in res) / n, 4), new_first=round(sum(r["first_key"].startswith("new") for r in res) / n, 4),
                     applied=round(sum(r["outcome"] == "applied" for r in res) / n, 4), outcomes=dict(collections.Counter(r["outcome"] for r in res)),
                     mean_tokens=round(sum(r["tokens"] or 0 for r in res) / n, 1))


def order_summary(rows, n_boot=10000, seed=0):
    """Apply rates by order, a paired test on seed 0, and a bootstrap over tasks, because states from one task are related."""
    x = [r["old_first"] == "applied" for r in rows if r["seed"] == 0]
    y = [r["new_first"] == "applied" for r in rows if r["seed"] == 0]
    b, c = sum(a and not d for a, d in zip(x, y)), sum(d and not a for a, d in zip(x, y))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["instance_id"]].append((r["old_first"] == "applied") - (r["new_first"] == "applied"))
    tasks = sorted(by)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        v = [d for t in rng.choices(tasks, k=len(tasks)) for d in by[t]]
        boots.append(sum(v) / max(1, len(v)))
    boots.sort()
    count = lambda k, key: sum(r[k] == key for r in rows)
    return dict(pairs=len(rows), tasks=len(tasks), rate_old_first=round(count("old_first", "applied") / max(1, len(rows)), 4),
                rate_new_first=round(count("new_first", "applied") / max(1, len(rows)), 4), seed0_only_old_first=b, seed0_only_new_first=c,
                mcnemar_p_seed0=mcnemar_exact(b, c), mean_gain=round(sum(sum(v) for v in by.values()) / max(1, len(rows)), 4),
                task_bootstrap_ci=[round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)],
                outcomes_old_first=dict(collections.Counter(r["old_first"] for r in rows)), outcomes_new_first=dict(collections.Counter(r["new_first"] for r in rows)))


def run(server, tok, cases, lora_scales=None, seeds=(0,), temperature=0.2, top_p=0.95, extra=None, prompt_fn=None, budget=THINK_BUDGET):
    """Replays each edit state. Failed-edit states get every seed, applied states one, so both are measured at low cost."""
    if SMOKE:
        seeds, budget = seeds[:1], min(budget, 128)
    groups, skipped = [], []
    for c in cases:
        p = (prompt_fn or render)(tok, c)
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        if n + 1024 > SLOT_CTX:
            skipped.append(c["case"])
            continue
        groups.append((c, p, n, seeds[:1] if c["ok"] else seeds))

    def one(g):
        c, p, n, sds = g
        rows = []
        for sd in sds:  # one seed after another, so the slot reuses its prompt cache
            sampling = dict(temperature=temperature, top_p=top_p, top_k=0, seed=sd, stop=STOP, cache_prompt=True, **(extra or {}))
            r = generate(server, p, n, sampling, lora_scales, budget)
            text = r.pop("text")
            rows.append(dict(case=c["case"], original=c["kind"], seed=sd, n_prompt=n, **r, **score(text, c), loop=loop_stats(text)["loop"]))
        return rows
    with ThreadPoolExecutor(server.parallel) as ex:
        res = [r for rows in ex.map(one, groups) for r in rows]

    def stats(v):
        edits = [r for r in v if r["tool"] == "edit_file"]
        ok = sum(r["outcome"] == "applied" for r in edits)
        return dict(n=len(v), edit_calls=len(edits), applied=ok, apply_rate=round(ok / max(1, len(edits)), 4), not_found=sum(r["outcome"] == "not_found" for r in edits),
                    identical=sum(r["outcome"] == "identical" for r in edits), multiple=sum(r["outcome"] == "multiple" for r in edits), backslash_edits=sum(r["backslash"] for r in edits),
                    other_tool=sum(r["outcome"] == "other_tool" for r in v), no_tool=sum(r["outcome"] == "no_tool" for r in v), loops=sum(r["loop"] for r in v))
    gen = [r["n_gen"] or 0 for r in res]
    return dict(skipped_too_long=skipped, after_failed_edit_state=stats([r for r in res if r["original"] != "applied"]),
                after_applied_edit_state=stats([r for r in res if r["original"] == "applied"]), all=stats(res), mean_tokens=round(sum(gen) / max(1, len(gen)), 1)), res

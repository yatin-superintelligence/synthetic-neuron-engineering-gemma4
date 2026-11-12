"""SWE-bench agent: Gemma 4 with the competition's tool set, Gemma's native tool-call syntax, official scoring.

Model: llama-server on the Arc GPU (native Windows). Repositories and tests: SWE-bench's published Docker images,
CPU only, no network. One container for the agent; a fresh container for scoring, which applies the agent's patch and
runs the official eval_script, then the official log parser. A prediction is resolved when all FAIL_TO_PASS and all
PASS_TO_PASS tests pass (same rule as swebench.harness.grading).

Because a small model may resolve few issues, every task also records graded metrics with more statistical power:
tool-call validity, reaching and editing the gold file, edit_file old_string failures (copy fidelity), patch produced,
patch applies, fraction of FAIL_TO_PASS passing, repeated-action loops.
"""
import ast
import json
import os
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import DATA
from .engine import strip_bos

SYSTEM = ("You are an autonomous software engineering agent working in a repository at /testbed. Fix the issue by "
          "editing the source code. First locate the relevant code with run_command (grep, ls) and read_file, then make "
          "the smallest correct change with edit_file, optionally run a test, then call submit_patch. Do not edit tests.")


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        "required": req}}}


TOOLS = [
    _fn("run_command", "Execute a shell command with /bin/bash -c inside /testbed.",
        {"command": ("string", "the shell command")}, ["command"]),
    _fn("read_file", "Read a file with 1-indexed inclusive line slicing.",
        {"filepath": ("string", "path relative to /testbed"), "start_line": ("integer", "first line"),
         "end_line": ("integer", "last line")}, ["filepath"]),
    _fn("edit_file", "Replace old_string with new_string in an existing file. old_string must match exactly and "
        "occur exactly once.", {"filepath": ("string", "path"), "old_string": ("string", "exact existing text"),
                                "new_string": ("string", "replacement text")}, ["filepath", "old_string", "new_string"]),
    _fn("write_file", "Create or overwrite a file.", {"filepath": ("string", "path"),
                                                      "content": ("string", "full content")}, ["filepath", "content"]),
    _fn("get_status", "Return remaining budget and patch status.", {}, []),
    _fn("submit_patch", "Finish: capture the current git diff as the final patch.", {}, []),
]

CALL = re.compile(r"<\|tool_call>call:([A-Za-z_]+)\{(.*?)\}<tool_call\|>", re.S)
SQ = '<|"|>'


def parse_args(s):
    """Parses Gemma's call syntax: key:<|"|>text<|"|>,key2:123"""
    args, i, n = {}, 0, len(s)
    while i < n:
        j = s.find(":", i)
        if j < 0:
            break
        key = s[i:j].strip().strip(",").strip()
        i = j + 1
        if s.startswith(SQ, i):
            k = s.find(SQ, i + len(SQ))
            if k < 0:
                return None
            args[key] = s[i + len(SQ):k]
            i = k + len(SQ)
        else:
            k = s.find(",", i)
            k = n if k < 0 else k
            v = s[i:k].strip()
            try:
                args[key] = json.loads(v)
            except Exception:  # noqa: BLE001
                args[key] = v
            i = k
        if i < n and s[i] == ",":
            i += 1
    return args


class Sandbox:
    """One Docker container. workdir/conda differ between SWE-bench (/testbed, conda env) and the competition
    sandbox (/workspace, system Python)."""

    def __init__(self, image, cpus=4, mem="4g", workdir="/testbed", conda=True):
        self.image, self.cpus, self.mem, self.workdir, self.conda = image, cpus, mem, workdir, conda
        self.name = f"sne_{uuid.uuid4().hex[:10]}"

    def __enter__(self):
        subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name, "--network", "none", "--cpus", str(self.cpus),
                        "--memory", self.mem, "--platform", "linux/amd64", "-w", self.workdir, self.image,
                        "sleep", "infinity"], check=True, capture_output=True, text=True)
        return self

    def __exit__(self, *a):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, text=True)

    def run(self, cmd, timeout=120, stdin=None, env_setup=True):
        use = env_setup and self.conda
        pre = "source /opt/miniconda3/bin/activate >/dev/null 2>&1; conda activate testbed >/dev/null 2>&1; " if use else ""
        try:
            p = subprocess.run(["docker", "exec", "-i", "-w", self.workdir, self.name, "bash", "-c", pre + cmd],
                               input=stdin, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                               errors="replace")
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {timeout} s."

    def write(self, path, content):
        return self.run(f"mkdir -p \"$(dirname '{path}')\" && cat > '{path}'", stdin=content, env_setup=False)

    def read(self, path):
        return self.run(f"cat '{path}'", env_setup=False)

    def diff(self):
        self.run("git add -A", env_setup=False)
        return self.run("git diff --cached HEAD", env_setup=False)[1]


def clip(text, limit=5000):
    if len(text) <= limit:
        return text
    h = limit * 2 // 3
    return text[:h] + f"\n... [{len(text) - limit} characters omitted] ...\n" + text[-(limit - h):]


def apply_replacement(content, old, new):
    """Three-tier matching as documented for the competition's edit_file: exact, line-stripped flexible, regex over
    code delimiters. Returns (new_content, strategy, n_matches); new_content is None unless exactly one match."""
    if not old:
        return None, "empty", 0
    n = content.count(old)
    if n == 1:
        return content.replace(old, new, 1), "exact", 1
    if n > 1:
        return None, "exact", n
    src, pat = content.split("\n"), [l.strip() for l in old.strip("\n").split("\n")]
    hits = [i for i in range(len(src) - len(pat) + 1) if [l.strip() for l in src[i:i + len(pat)]] == pat]
    if len(hits) == 1:
        i = hits[0]
        ind = src[i][: len(src[i]) - len(src[i].lstrip())]
        nl = new.strip("\n").split("\n")
        base = len(nl[0]) - len(nl[0].lstrip()) if nl else 0
        fixed = [ind + l[base:] if l.strip() else l for l in nl]
        return "\n".join(src[:i] + fixed + src[i + len(pat):]), "flexible", 1
    if len(hits) > 1:
        return None, "flexible", len(hits)
    toks = [re.escape(t) for t in re.split(r"(\(|\)|:|\[|\]|\{|\}|>|=|<)|\s+", old) if t]
    if not toks:
        return None, "none", 0
    rx = re.compile(r"\s*".join(toks))
    ms = list(rx.finditer(content))
    if len(ms) == 1:
        m = ms[0]
        return content[:m.start()] + new + content[m.end():], "regex", 1
    return None, "regex" if ms else "none", len(ms)


class Episode:
    def __init__(self, task, max_turns):
        self.t, self.max_turns = task, max_turns
        self.m = dict(turns=0, tool_calls=0, malformed=0, edit_ok=0, edit_fail_notfound=0, edit_fail_multiple=0,
                      edit_exact=0, edit_flexible=0, edit_regex=0, read_calls=0, cmd_calls=0, write_calls=0,
                      submitted=False, loop_abort=False, no_call_end=False, gen_tokens=0, truncated_turns=0)
        self.seen = set()
        self.gold = set(re.findall(r"^diff --git a/(\S+)", task["patch"], re.M))
        self.touched_gold = False
        self.last_calls = []

    def tool(self, sb, name, a):
        m = self.m
        for g in self.gold:
            if g in json.dumps(a):
                self.touched_gold = True
        if name == "run_command":
            m["cmd_calls"] += 1
            rc, out = sb.run(a.get("command", ""), timeout=120)
            if any(g in out for g in self.gold):
                self.touched_gold = True
            return clip(out) or f"(no output, exit code {rc})"
        if name == "read_file":
            m["read_calls"] += 1
            rc, out = sb.read(a.get("filepath", ""))
            if rc != 0:
                return f"Error: cannot read {a.get('filepath')}"
            lines = out.split("\n")
            try:
                s = int(a.get("start_line") or 1)
                e = int(a.get("end_line") or s + 149)
            except (TypeError, ValueError):
                s, e = 1, 150
            e = min(e, s + 149, len(lines))
            return clip("\n".join(lines[max(0, s - 1):e]), 10000) + f"\n[lines {s}-{e} of {len(lines)}]"
        if name == "edit_file":
            rc, out = sb.read(a.get("filepath", ""))
            if rc != 0 or not out:
                m["edit_fail_notfound"] += 1
                return f"Error: file {a.get('filepath')} does not exist or is empty."
            new, strategy, n = apply_replacement(out.replace("\r\n", "\n"), a.get("old_string", ""),
                                                 str(a.get("new_string", "")))
            if new is None and n == 0:
                m["edit_fail_notfound"] += 1
                return "Error: old_string not found in the file."
            if new is None:
                m["edit_fail_multiple"] += 1
                return f"Error: old_string matches {n} locations; include more context so it is unique."
            sb.write(a["filepath"], new)
            m["edit_ok"] += 1
            m[f"edit_{strategy}"] += 1
            return f"Edited {a['filepath']} (match: {strategy})."
        if name == "write_file":
            m["write_calls"] += 1
            sb.write(a.get("filepath", ""), a.get("content", ""))
            return f"Wrote {a.get('filepath')}."
        if name == "get_status":
            return f"Turn {m['turns']} of {self.max_turns}. Current diff lines: {len(sb.diff().splitlines())}."
        if name == "submit_patch":
            m["submitted"] = True
            return "Patch submitted."
        return f"Error: unknown tool {name}."


def _render(tok, msgs):
    return strip_bos(tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True))


def _fit(tok, msgs, budget):
    """Context compaction: elide the oldest tool outputs until the prompt fits."""
    while True:
        s = _render(tok, msgs)
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if n <= budget:
            return s, n
        for m in msgs:
            if m["role"] == "tool" and m["content"] != "[output elided]":
                m["content"] = "[output elided]"
                break
        else:
            return s, n


def run_agent(server, tok, task, max_turns=100, max_tokens=8192, ctx_budget=24000, lora_scales=None, wall_limit=3600,
              sandbox_factory=None, system=None, user=None):
    """sandbox_factory(task) -> context manager yielding a ready Sandbox. Defaults to the SWE-bench image."""
    ep = Episode(task, max_turns)
    msgs = [{"role": "system", "content": system or SYSTEM},
            {"role": "user", "content": user or f"Issue:\n{task['problem_statement']}\n\nFix this issue in /testbed."}]
    t0 = time.time()
    factory = sandbox_factory or (lambda t: Sandbox(t["image"]))
    with factory(task) as sb:
        for _ in range(max_turns):
            if time.time() - t0 > wall_limit:
                break
            ep.m["turns"] += 1
            prompt, _n = _fit(tok, msgs, ctx_budget)
            g = server.complete(prompt, max_tokens=max_tokens, lora_scales=lora_scales, sampling=dict(cache_prompt=True))
            ep.m["gen_tokens"] += g["n_gen"] or 0
            ep.m["truncated_turns"] += g["stop_type"] == "limit"
            text = g["text"]
            calls = CALL.findall(text)
            if not calls:
                ep.m["malformed"] += "<|tool_call>" in text
                msgs.append({"role": "assistant", "content": re.sub(r"<[|/]?[a-z_]+\|?>", "", text)[:2000]})
                if ep.m.get("_nudged"):
                    ep.m["no_call_end"] = True
                    break
                ep.m["_nudged"] = 1
                msgs.append({"role": "user", "content": "Continue. Use a tool call; call submit_patch when finished."})
                continue
            tcs, results = [], []
            for name, raw in calls[:4]:
                a = parse_args(raw)
                if a is None:
                    ep.m["malformed"] += 1
                    a = {}
                ep.m["tool_calls"] += 1
                tcs.append({"type": "function", "function": {"name": name, "arguments": a}})
                results.append((name, ep.tool(sb, name, a)))
            msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
            for name, r in results:
                msgs.append({"role": "tool", "name": name, "content": r})
            sig = json.dumps(tcs, sort_keys=True)
            ep.last_calls.append(sig)
            if len(ep.last_calls) >= 4 and len(set(ep.last_calls[-4:])) == 1:
                ep.m["loop_abort"] = True
                break
            if ep.m["submitted"]:
                break
        patch = sb.diff()
    ep.m.update(wall_s=round(time.time() - t0, 1), visited_gold=ep.touched_gold,
                patch_nonempty=bool(patch.strip()), patch_lines=len(patch.splitlines()),
                edited_gold=any(f"a/{g} " in patch for g in ep.gold))
    return patch, ep.m, msgs


def tests(v):
    return v if isinstance(v, list) else ast.literal_eval(v)


def score(task, patch):
    """Official-rule scoring in a fresh container. Returns dict with resolved and graded metrics."""
    from swebench.harness import log_parsers as lp
    from swebench.harness.grading import test_passed
    f2p, p2p = tests(task["FAIL_TO_PASS"]), tests(task["PASS_TO_PASS"])
    if not patch.strip():
        return dict(resolved=False, applied=False, f2p_passed=0, f2p_total=len(f2p), p2p_passed=0, p2p_total=len(p2p),
                    note="empty patch")
    with Sandbox(task["image"], cpus=4) as sb:
        sb.write("/tmp/agent.patch", patch if patch.endswith("\n") else patch + "\n")
        rc, out = sb.run("git apply --verbose --reject /tmp/agent.patch || patch --batch --fuzz=5 -p1 -i /tmp/agent.patch",
                         env_setup=False)
        applied = rc == 0
        if not applied:
            return dict(resolved=False, applied=False, f2p_passed=0, f2p_total=len(f2p), p2p_passed=0,
                        p2p_total=len(p2p), note=clip(out, 600))
        sb.write("/tmp/eval.sh", task["eval_script"])
        rc, log = sb.run("bash /tmp/eval.sh", timeout=1500, env_setup=False)
    parser = getattr(lp, task["log_parser"])
    try:
        sm = parser(log, None)
    except TypeError:
        sm = parser(log)
    fp = sum(test_passed(c, sm) for c in f2p)
    pp = sum(test_passed(c, sm) for c in p2p)
    return dict(resolved=fp == len(f2p) and pp == len(p2p), applied=True, f2p_passed=fp, f2p_total=len(f2p),
                p2p_passed=pp, p2p_total=len(p2p), log_tail=clip(log, 1200), n_parsed=len(sm))


def load_subset(split=None, path=None):
    """Rule-based subset, fixed before any agent run: SWE-bench Verified, difficulty '<15 min fix', single-file gold
    patch, excluding repos with very heavy images (matplotlib, scikit-learn, astropy, xarray). Seeded dev/test split."""
    import random
    path = path or os.path.join(DATA, "swebench_verified.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    skip = {"matplotlib/matplotlib", "scikit-learn/scikit-learn", "astropy/astropy", "pydata/xarray", "mwaskom/seaborn"}
    sel = [r for r in rows if r["difficulty"] == "<15 min fix" and r["repo"] not in skip
           and len(re.findall(r"^diff --git a/", r["patch"], re.M)) == 1]
    ids = sorted(r["instance_id"] for r in sel)
    random.Random(20261106).shuffle(ids)
    dev = set(ids[:60])
    out = sel if split is None else [r for r in sel if (r["instance_id"] in dev) == (split == "dev")]
    from . import SMOKE
    return out[:2] if SMOKE else out


def run_many(server, tok, tasks, workers=4, lora_scales=None, max_turns=100, do_score=True, log=print, scorer=None,
             sandbox_factory=None, system=None, user_fn=None, ctx_budget=24000):
    scorer = scorer or score

    def one(t):
        try:
            patch, m, msgs = run_agent(server, tok, t, lora_scales=lora_scales, max_turns=max_turns,
                                       sandbox_factory=sandbox_factory, system=system, ctx_budget=ctx_budget,
                                       user=user_fn(t) if user_fn else None)
            s = scorer(t, patch) if do_score else {}
            return dict(instance_id=t["instance_id"], repo=t["repo"], patch=patch, **m, **s, conversation=msgs)
        except Exception as e:  # noqa: BLE001
            return dict(instance_id=t["instance_id"], repo=t["repo"], error=repr(e))
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, tasks):
            rows.append(r)
            log(r["instance_id"], {k: r.get(k) for k in ("turns", "tool_calls", "malformed", "edit_ok", "edit_fail_notfound",
                                                         "visited_gold", "patch_nonempty", "applied", "f2p_passed",
                                                         "f2p_total", "resolved", "error")})
    return rows


def summarize(rows):
    ok = [r for r in rows if "error" not in r]
    n = max(1, len(ok))
    keys = ["visited_gold", "edited_gold", "patch_nonempty", "applied", "submitted", "loop_abort", "resolved"]
    s = {k: round(sum(bool(r.get(k)) for r in ok) / n, 4) for k in keys}
    s.update(n=len(ok), errors=len(rows) - len(ok), mean_turns=round(sum(r["turns"] for r in ok) / n, 2),
             malformed_per_task=round(sum(r["malformed"] for r in ok) / n, 3),
             edit_ok=sum(r["edit_ok"] for r in ok), edit_fail_notfound=sum(r["edit_fail_notfound"] for r in ok),
             edit_fail_multiple=sum(r["edit_fail_multiple"] for r in ok),
             f2p_frac=round(sum(r.get("f2p_passed", 0) / max(1, r.get("f2p_total", 1)) for r in ok) / n, 4),
             edit_exact=sum(r.get("edit_exact", 0) for r in ok), edit_flexible=sum(r.get("edit_flexible", 0) for r in ok),
             edit_regex=sum(r.get("edit_regex", 0) for r in ok),
             gen_tokens=sum(r["gen_tokens"] for r in ok))
    e = s["edit_ok"] + s["edit_fail_notfound"]
    s["edit_success_rate"] = round(s["edit_ok"] / e, 4) if e else None
    return s


def ensure_docker(log=print):
    """Starts Docker Desktop only while repository tasks run (it is CPU-only sandboxing; the model stays on the GPU)."""
    if subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True).returncode == 0:
        return
    subprocess.Popen(["powershell", "-NoProfile", "-Command",
                      "Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'"])
    for _ in range(40):
        time.sleep(5)
        if subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True).returncode == 0:
            log("docker engine up")
            return
    raise RuntimeError("docker engine did not start")


def stop_docker():
    subprocess.run(["docker", "desktop", "stop"], capture_output=True)


def pull(tasks, log=print):
    for t in tasks:
        have = subprocess.run(["docker", "image", "inspect", t["image"]], capture_output=True).returncode == 0
        if not have:
            t0 = time.time()
            subprocess.run(["docker", "pull", "-q", "--platform", "linux/amd64", t["image"]], check=True, capture_output=True)
            log(f"pulled {t['image']} in {time.time() - t0:.0f}s")

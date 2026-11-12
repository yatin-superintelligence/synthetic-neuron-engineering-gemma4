"""Experiment registry: every experiment gets experiments/EXP-###_slug/ with README.md, run.log, config.json,
results.json (+ raw outputs), and a row in experiments/INDEX.md. Provenance (git commit, env, timings) is automatic.
"""
import contextlib
import datetime as dt
import io
import json
import os
import platform
import subprocess
import sys
import time

from . import ROOT, SMOKE

EXP_DIR = os.path.join(ROOT, "scratch", "preflight") if SMOKE else os.path.join(ROOT, "experiments")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def env_info():
    info = dict(python=sys.version.split()[0], platform=platform.platform(), git_commit=_git("rev-parse", "HEAD"),
                git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")))
    try:
        import torch
        import transformers
        info.update(torch=torch.__version__, transformers=transformers.__version__,
                    device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else torch.xpu.get_device_name(0) if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    except Exception:  # noqa: BLE001
        pass
    return info


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


class Experiment:
    """Usage:
        with Experiment("EXP-012", "ablate-top-code-neurons", motivation=..., hypothesis=..., config={...}) as ex:
            ... ex.log(...); ex.save_json("generations.json", obj); ex.result(key=value) ...
        README.md is rendered on exit (also on failure, marked FAILED).
    """

    def __init__(self, exp_id, slug, motivation, hypothesis, config, depends_on=(), category=""):
        self.id, self.slug = exp_id, slug
        self.dir = os.path.join(EXP_DIR, f"{exp_id}_{slug}")
        os.makedirs(self.dir, exist_ok=True)
        self.meta = dict(id=exp_id, slug=slug, category=category, motivation=motivation, hypothesis=hypothesis,
                         config=config, depends_on=list(depends_on), env=env_info(),
                         command=" ".join(sys.argv), started=dt.datetime.utcnow().isoformat() + "Z")
        self.results = {}
        self.notes = []

    def __enter__(self):
        self._logf = open(os.path.join(self.dir, "run.log"), "a", encoding="utf-8")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(self._stdout, self._logf)
        sys.stderr = Tee(self._stderr, self._logf)
        self._t0 = time.time()
        # keep the previous results so an interrupted experiment can resume finished suites
        prev = os.path.join(self.dir, "results.json")
        if os.path.exists(prev):
            import shutil
            shutil.copy(prev, os.path.join(self.dir, "results_resume.json"))
        from .sysmon import Sampler, guard
        self.meta["sys_at_start"] = guard(log=print)
        self._sampler = Sampler(os.path.join(self.dir, "sysmon.jsonl"), every=60).start()
        print(f"=== {self.id} {self.slug} started {self.meta['started']} commit={self.meta['env'].get('git_commit','')[:10]}")
        with open(os.path.join(self.dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        # exact code provenance: the script itself + any uncommitted changes to the library
        try:
            import shutil
            shutil.copy(os.path.abspath(sys.argv[0]), os.path.join(self.dir, "script_snapshot.py"))
            diff = _git("diff", "HEAD", "--", "src", "experiments/scripts")
            if diff:
                with open(os.path.join(self.dir, "uncommitted.diff"), "w", encoding="utf-8") as f:
                    f.write(diff)
        except Exception as e:  # noqa: BLE001
            print("provenance snapshot failed:", e)
        return self

    def log(self, *a):
        print(f"[{dt.datetime.utcnow().strftime('%H:%M:%S')}]", *a)

    def note(self, text):
        self.notes.append(text)

    def result(self, **kw):
        self.results.update(kw)
        self._dump_results()

    def save_json(self, name, obj):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=str)

    def _dump_results(self):
        with open(os.path.join(self.dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(dict(results=self.results, wall_seconds=time.time() - self._t0), f, indent=2, default=str)

    def __exit__(self, et, ev, tb):
        wall = time.time() - self._t0
        from .sysmon import summarize
        self._sampler.stop()
        self.results["system_monitor"] = summarize(os.path.join(self.dir, "sysmon.jsonl"))
        status = "OK" if et is None or (et is SystemExit and not ev.code) else f"FAILED: {et.__name__}: {ev}"
        print(f"=== {self.id} finished status={status} wall={wall:.1f}s")
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._logf.close()
        self.meta.update(finished=dt.datetime.utcnow().isoformat() + "Z", wall_seconds=round(wall, 1), status=status)
        with open(os.path.join(self.dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        self._dump_results()
        self._readme(status, wall)
        rebuild_index()
        return False

    def _readme(self, status, wall):
        m = self.meta
        cfg = json.dumps(m["config"], indent=2, default=str)
        res = json.dumps(self.results, indent=2, default=str)
        lines = [f"# {self.id}: {self.slug}", "",
                 f"- **Status:** {status}", f"- **Category:** {m['category']}",
                 f"- **Started / finished (UTC):** {m['started']} / {m.get('finished')}",
                 f"- **Wall time:** {wall/60:.1f} min",
                 f"- **Commit:** `{m['env'].get('git_commit')}` (dirty={m['env'].get('git_dirty')})",
                 f"- **Depends on:** {', '.join(m['depends_on']) or '-'}",
                 f"- **Command:** `{m['command']}`", "",
                 "## Why this experiment", m["motivation"], "",
                 "## Hypothesis (stated before running)", m["hypothesis"], "",
                 "## Configuration", "```json", cfg, "```", "",
                 "## Results", "```json", res, "```", ""]
        if self.notes:
            lines += ["## Notes / interpretation", *[f"- {n}" for n in self.notes], ""]
        lines += ["## Files", "- `script_snapshot.py` exact script that produced this folder (+ `uncommitted.diff` if any)",
                  "- `run.log` full console log", "- `config.json` config + environment provenance",
                  "- `results.json` metrics", "- `sysmon.jsonl` RAM/GPU/battery/thermal samples every 60 s", "- other `*.json` raw outputs (generations, per-problem outcomes)"]
        with open(os.path.join(self.dir, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def rebuild_index():
    """INDEX.md: one row per experiment folder, numbered by execution order (Run #) so the count of experiments
    actually run is explicit. Experiment IDs are block-allocated by category (see PLAN.md), so they have gaps."""
    os.makedirs(EXP_DIR, exist_ok=True)
    recs = []
    for d in sorted(os.listdir(EXP_DIR)):
        p = os.path.join(EXP_DIR, d, "config.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        r = {}
        rp = os.path.join(EXP_DIR, d, "results.json")
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as f:
                r = json.load(f).get("results", {})
        recs.append((m.get("started", ""), d, m, r))
    recs.sort()
    rows = []
    done = 0
    for n, (st, d, m, r) in enumerate(recs, 1):
        status = m.get("status", "running")
        done += status == "OK"
        rows.append(f"| {n} | [{m['id']}]({d}/README.md) | {m.get('category','')} | {m['slug']} | {status} | "
                    f"{st[:16].replace('T', ' ')} | {m.get('wall_seconds','')} | {r.get('headline', '')} |")
    failed = sorted(os.listdir(os.path.join(EXP_DIR, "_failed"))) if os.path.isdir(os.path.join(EXP_DIR, "_failed")) else []
    with open(os.path.join(EXP_DIR, "INDEX.md"), "w", encoding="utf-8") as f:
        header = ("# Experiment index\n\nAuto-generated by `sne.registry.rebuild_index()`. Run # is "
                  "execution order. Experiment IDs are allocated in blocks by category (see PLAN.md), so gaps in "
                  "IDs are expected.\n\n")
        header += (f"Completed OK: {done}. Folders: {len(recs)}. Archived failed attempts: {len(failed)} "
                   f"({', '.join(failed) or '-'}).\n\n")
        header += ("| Run # | ID | Category | Experiment | Status | Started (UTC) | Wall (s) | Headline |\n"
                   "|---|---|---|---|---|---|---|---|\n")
        f.write(header + "\n".join(rows) + "\n")

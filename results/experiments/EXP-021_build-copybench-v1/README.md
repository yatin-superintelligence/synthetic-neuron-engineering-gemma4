# EXP-021: build-copybench-v1

- **Status:** OK
- **Category:** benchmark
- **Started / finished (UTC):** 2026-11-05T20:15:45.241696Z / 2026-11-05T20:16:06.736952Z
- **Wall time:** 0.2 min
- **Commit:** `cfeb1dee07b8391fc0ee8a5336a60ebbaeff3ab7` (dirty=True)
- **Depends on:** -
- **Command:** `experiments/scripts/exp021_build_copybench.py`

## Why this experiment
H-1 needs a cheap, high-power measure of exact copying from context, the skill behind edit_file old_string, paths and required identifiers. Real stdlib code keeps it disjoint from every other benchmark. Short outputs allow screening many edits.

## Hypothesis (stated before running)
All oracle answers pass the judge and trivially wrong answers fail.

## Configuration
```json
{
  "seed": 20261107,
  "modules": [
    "argparse",
    "csv",
    "gettext",
    "ipaddress",
    "json/decoder",
    "logging/__init__",
    "mailbox",
    "mimetypes",
    "optparse",
    "pathlib",
    "pprint",
    "shlex",
    "smtplib",
    "string",
    "tarfile",
    "textwrap",
    "urllib/parse",
    "uuid",
    "zipfile/__init__",
    "configparser",
    "difflib",
    "dataclasses",
    "fnmatch",
    "glob",
    "heapq",
    "html/parser",
    "http/cookies",
    "statistics",
    "tokenize",
    "calendar"
  ],
  "per_kind": 100
}
```

## Results
```json
{
  "n": 300,
  "kinds": {
    "name": 100,
    "edit": 100,
    "verbatim": 100
  },
  "splits": {
    "test": 150,
    "dev": 150
  },
  "oracle_pass": {
    "name": 100,
    "edit": 100,
    "verbatim": 100
  },
  "wrong_pass": {
    "name": 0,
    "edit": 0,
    "verbatim": 0
  },
  "headline": "300 tasks; oracle pass {'name': 100, 'edit': 100, 'verbatim': 100}; wrong pass {'name': 0, 'edit': 0, 'verbatim': 0}",
  "system_monitor": {
    "samples": 1,
    "ram_free_gb": {
      "min": 6.14,
      "max": 6.14,
      "mean": 6.14
    },
    "gpu_shared_gb": {
      "min": 7.04,
      "max": 7.04,
      "mean": 7.04
    },
    "gpu_util_pct": {
      "min": 90.3,
      "max": 90.3,
      "mean": 90.3
    },
    "thermal_c": {
      "min": 27.9,
      "max": 27.9,
      "mean": 27.9
    },
    "battery_pct": {
      "min": 56,
      "max": 56,
      "mean": 56.0
    },
    "cpu_pct": {
      "min": 15.2,
      "max": 15.2,
      "mean": 15.2
    },
    "cpu_perf_pct": {
      "min": 95.7,
      "max": 95.7,
      "mean": 95.7
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_freq_mhz": {
      "min": 1867,
      "max": 1867,
      "mean": 1867.0
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 0,
      "mean": 0.0
    },
    "battery_charge_mw": {
      "min": 35918,
      "max": 35918,
      "mean": 35918.0
    }
  }
}
```

## Files
- `script_snapshot.py` exact script that produced this folder (+ `uncommitted.diff` if any)
- `run.log` full console log
- `config.json` config + environment provenance
- `results.json` metrics
- `sysmon.jsonl` RAM/GPU/battery/thermal samples every 60 s
- other `*.json` raw outputs (generations, per-problem outcomes)

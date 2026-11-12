# EXP-020: build-repopatch-v1

- **Status:** OK
- **Category:** benchmark
- **Started / finished (UTC):** 2026-11-05T19:32:27.548006Z / 2026-11-05T19:34:00.021109Z
- **Wall time:** 1.5 min
- **Commit:** `7afff233ed9c72868c6863967b48d048597ca94d` (dirty=True)
- **Depends on:** EXP-002
- **Command:** `experiments/scripts/exp020_build_repopatch.py`

## Why this experiment
The competition scores agents on resolving real issues by editing files through tool calls. Function-completion benchmarks miss three skills a small agent needs: locating the faulty code among distractors, reading an issue report, and emitting well-formed edit calls whose old_string matches exactly. RepoPatch measures these, cheaply and executably.

## Hypothesis (stated before running)
Every kept task is valid: the oracle edit resolves it and the unedited repo fails.

## Configuration
```json
{
  "n_target": 150,
  "seed": 20261106,
  "mutation_ops": [
    "cmp",
    "const",
    "arith",
    "bool",
    "not"
  ],
  "files": [
    "pkg/core.py",
    "pkg/utils.py"
  ],
  "distractors": 4
}
```

## Results
```json
{
  "n_tasks": 141,
  "mutations": {
    "off-by-one integer constant": 48,
    "boolean operator swapped": 7,
    "arithmetic operator swapped": 40,
    "comparison operator flipped": 46
  },
  "oracle_resolves": 141,
  "buggy_passes": 0,
  "prompt_chars": {
    "min": 1583,
    "max": 3863,
    "mean": 2327.4609929078015
  },
  "path": "<repo>\\data\\repopatch_v1.jsonl",
  "headline": "141 valid tasks; mutations {'off-by-one integer constant': 48, 'boolean operator swapped': 7, 'arithmetic operator swapped': 40, 'comparison operator flipped': 46}",
  "system_monitor": {
    "samples": 2,
    "ram_free_gb": {
      "min": 12.65,
      "max": 14.28,
      "mean": 13.46
    },
    "gpu_shared_gb": {
      "min": 3.57,
      "max": 5,
      "mean": 4.29
    },
    "gpu_util_pct": {
      "min": 87.2,
      "max": 135.9,
      "mean": 111.55
    },
    "thermal_c": {
      "min": 27.9,
      "max": 27.9,
      "mean": 27.9
    },
    "battery_pct": {
      "min": 39,
      "max": 39,
      "mean": 39.0
    },
    "cpu_pct": {
      "min": 24.8,
      "max": 55.2,
      "mean": 40.0
    }
  }
}
```

## Files
- `run.log` full console log
- `config.json` config + environment provenance
- `results.json` metrics
- `sysmon.jsonl` RAM/GPU/battery/thermal samples every 60 s
- other `*.json` raw outputs (generations, per-problem outcomes)

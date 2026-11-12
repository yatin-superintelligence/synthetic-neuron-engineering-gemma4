# EXP-105: instruction-editbench-f16

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-09T12:29:23.217409Z / 2026-11-09T16:12:31.953903Z
- **Wall time:** 223.1 min
- **Commit:** `f28ddf20b3f2fb58759c77fd180b37676706c1b4` (dirty=True)
- **Depends on:** EXP-097b, EXP-099b
- **Command:** `experiments/scripts/exp105_instruction_editbench.py`

## Why this experiment
The one bench result outside the noise floor is an instruction (D-102); does it raise applied edits on real official-run states?

## Hypothesis (stated before running)
On F16 EditBench the instruction raises applied edit calls and lowers backslash-carrying ones against the unedited model and against a neutral instruction of the same length (exact McNemar by state).

## Configuration
```json
{
  "max_prompt": 12000,
  "per_run_task": 1,
  "seed": 0,
  "precision": "f16",
  "texts": {
    "instruction": "The bug is one small mistake on one line. Once you have read the code that contains it, fix it right away with edit_file. Copy old_string character for character from the file content you read, without adding escape characters, and in new_string change only the wrong token or operator. Do not rewrite or restructure code.",
    "neutral": "The repository is a Python project. Its files use four spaces for indentation and its tests are written with pytest. The tools listed above are the only ones available in this session, and every tool call returns its result as JSON. Paths are given relative to the repository root, which is also the working directory of every command you run."
  }
}
```

## Results
```json
{
  "arms": {
    "base": {
      "skipped_too_long": [],
      "after_failed_edit_state": {
        "n": 14,
        "edit_calls": 4,
        "applied": 2,
        "apply_rate": 0.5,
        "not_found": 2,
        "identical": 0,
        "multiple": 0,
        "backslash_edits": 2,
        "other_tool": 10,
        "no_tool": 0,
        "loops": 0
      },
      "after_applied_edit_state": {
        "n": 48,
        "edit_calls": 32,
        "applied": 31,
        "apply_rate": 0.9688,
        "not_found": 0,
        "identical": 0,
        "multiple": 0,
        "backslash_edits": 6,
        "other_tool": 16,
        "no_tool": 0,
        "loops": 0
      },
      "all": {
        "n": 62,
        "edit_calls": 36,
        "applied": 33,
        "apply_rate": 0.9167,
        "not_found": 2,
        "identical": 0,
        "multiple": 0,
        "backslash_edits": 8,
        "other_tool": 26,
        "no_tool": 0,
        "loops": 0
      },
      "mean_tokens": 891.7
    },
    "instruction": {
      "skipped_too_long": [],
      "after_failed_edit_state": {
        "n": 14,
        "edit_calls": 7,
        "applied": 2,
        "apply_rate": 0.2857,
        "not_found": 5,
        "identical": 0,
        "multiple": 0,
        "backslash_edits": 3,
        "other_tool": 7,
        "no_tool": 0,
        "loops": 0
      },
      "after_applied_edit_state": {
        "n": 48,
        "edit_calls": 34,
        "applied": 31,
        "apply_rate": 0.9118,
        "not_found": 2,
        "identical": 1,
        "multiple": 0,
        "backslash_edits": 9,
        "other_tool": 14,
        "no_tool": 0,
        "loops": 4
      },
      "all": {
        "n": 62,
        "edit_calls": 41,
        "applied": 33,
        "apply_rate": 0.8049,
        "not_found": 7,
        "identical": 1,
        "multiple": 0,
        "backslash_edits": 12,
        "other_tool": 21,
        "no_tool": 0,
        "loops": 4
      },
      "mean_tokens": 1147.3
    },
    "neutral": {
      "skipped_too_long": [],
      "after_failed_edit_state": {
        "n": 14,
        "edit_calls": 7,
        "applied": 0,
        "apply_rate": 0.0,
        "not_found": 6,
        "identical": 0,
        "multiple": 0,
        "backslash_edits": 3,
        "other_tool": 7,
        "no_tool": 0,
        "loops": 1
      },
      "after_applied_edit_state": {
        "n": 48,
        "edit_calls": 32,
        "applied": 28,
        "apply_rate": 0.875,
        "not_found": 1,
        "identical": 0,
        "multiple": 1,
        "backslash_edits": 6,
        "other_tool": 16,
        "no_tool": 0,
        "loops": 0
      },
      "all": {
        "n": 62,
        "edit_calls": 39,
        "applied": 28,
        "apply_rate": 0.7179,
        "not_found": 7,
        "identical": 0,
        "multiple": 1,
        "backslash_edits": 9,
        "other_tool": 23,
        "no_tool": 0,
        "loops": 1
      },
      "mean_tokens": 986.4
    }
  },
  "states": 62,
  "totals": {
    "base": {
      "backslash": 8,
      "applied": 33,
      "edit_calls": 36
    },
    "instruction": {
      "backslash": 12,
      "applied": 33,
      "edit_calls": 41
    },
    "neutral": {
      "backslash": 9,
      "applied": 28,
      "edit_calls": 39
    }
  },
  "tests": {
    "instruction_vs_base_backslash": {
      "instruction_only": 5,
      "other_only": 1,
      "p": 0.2188
    },
    "instruction_vs_base_applied": {
      "instruction_only": 7,
      "other_only": 7,
      "p": 1.0
    },
    "instruction_vs_neutral_backslash": {
      "instruction_only": 8,
      "other_only": 5,
      "p": 0.5811
    },
    "instruction_vs_neutral_applied": {
      "instruction_only": 11,
      "other_only": 6,
      "p": 0.3323
    }
  },
  "headline": "F16 EditBench (62 states): applied unedited 33, instruction 33, neutral 28; backslash 8, 12, 9",
  "system_monitor": {
    "samples": 185,
    "ram_free_gb": {
      "min": 7.64,
      "max": 18.72,
      "mean": 8.26
    },
    "gpu_shared_gb": {
      "min": 1.16,
      "max": 6.84,
      "mean": 6.66
    },
    "gpu_util_pct": {
      "min": 5,
      "max": 100,
      "mean": 91.43
    },
    "cpu_package_w": {
      "min": 15.2,
      "max": 44.9,
      "mean": 24.07
    },
    "cpu_core_w": {
      "min": 1.2,
      "max": 24.6,
      "mean": 3.67
    },
    "igpu_w": {
      "min": 0,
      "max": 24.2,
      "mean": 12.8
    },
    "battery_pct": {
      "min": 92,
      "max": 100,
      "mean": 99.81
    },
    "cpu_pct": {
      "min": 0,
      "max": 100,
      "mean": 13.66
    },
    "cpu_perf_pct": {
      "min": 41.5,
      "max": 147.7,
      "mean": 68.59
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "passive_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "throttle_reasons": {
      "min": 0,
      "max": 0,
      "mean": 0.0
    },
    "cpu_freq_mhz": {
      "min": 1587,
      "max": 1918,
      "mean": 1809.79
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 35423,
      "mean": 1361.52
    },
    "battery_charge_mw": {
      "min": 0,
      "max": 0,
      "mean": 0.0
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

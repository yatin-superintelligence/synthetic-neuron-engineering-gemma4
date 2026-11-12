# EXP-060: loop-edit-on-edit-behaviour

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T06:24:29.269160Z / 2026-11-08T10:07:38.671604Z
- **Wall time:** 223.2 min
- **Commit:** `39c7d7540bb06767b2d4974a401fa7b2706d4593` (dirty=True)
- **Depends on:** EXP-059
- **Command:** `experiments/scripts/exp060_loop_edit_on_edits.py`

## Why this experiment
Perseveration and looping may share a cause. Exact copying is also what such an edit could damage.

## Hypothesis (stated before running)
The edit lowers same-text retries and leaves the apply rate on normal edit states unchanged.

## Configuration
```json
{
  "n_retry": 24,
  "n_edit": 16,
  "max_prompt": 16000,
  "seeds": [
    0
  ],
  "split": "dev"
}
```

## Results
```json
{
  "arms": {
    "q4_0": {
      "base": {
        "retry": {
          "n": 24,
          "same_text": 11,
          "applied": 0,
          "other_tool": 10,
          "no_tool": 1,
          "loops": 1,
          "mean_tokens": 556.7
        },
        "edit_states": {
          "skipped_too_long": [],
          "after_failed_edit_state": {
            "n": 4,
            "edit_calls": 4,
            "applied": 0,
            "apply_rate": 0.0,
            "not_found": 3,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 2,
            "other_tool": 0,
            "no_tool": 0,
            "loops": 0
          },
          "after_applied_edit_state": {
            "n": 12,
            "edit_calls": 11,
            "applied": 9,
            "apply_rate": 0.8182,
            "not_found": 1,
            "identical": 1,
            "multiple": 0,
            "backslash_edits": 0,
            "other_tool": 1,
            "no_tool": 0,
            "loops": 1
          },
          "all": {
            "n": 16,
            "edit_calls": 15,
            "applied": 9,
            "apply_rate": 0.6,
            "not_found": 4,
            "identical": 1,
            "multiple": 0,
            "backslash_edits": 2,
            "other_tool": 1,
            "no_tool": 0,
            "loops": 1
          },
          "mean_tokens": 1150.6
        }
      },
      "edit": {
        "retry": {
          "n": 24,
          "same_text": 5,
          "applied": 1,
          "other_tool": 13,
          "no_tool": 0,
          "loops": 0,
          "mean_tokens": 568.8
        },
        "edit_states": {
          "skipped_too_long": [],
          "after_failed_edit_state": {
            "n": 4,
            "edit_calls": 4,
            "applied": 1,
            "apply_rate": 0.25,
            "not_found": 2,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 2,
            "other_tool": 0,
            "no_tool": 0,
            "loops": 0
          },
          "after_applied_edit_state": {
            "n": 12,
            "edit_calls": 9,
            "applied": 6,
            "apply_rate": 0.6667,
            "not_found": 2,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 2,
            "other_tool": 2,
            "no_tool": 1,
            "loops": 1
          },
          "all": {
            "n": 16,
            "edit_calls": 13,
            "applied": 7,
            "apply_rate": 0.5385,
            "not_found": 4,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 4,
            "other_tool": 2,
            "no_tool": 1,
            "loops": 1
          },
          "mean_tokens": 2021.2
        }
      }
    },
    "f16": {
      "base": {
        "retry": {
          "n": 24,
          "same_text": 5,
          "applied": 1,
          "other_tool": 17,
          "no_tool": 0,
          "loops": 0,
          "mean_tokens": 338.5
        },
        "edit_states": {
          "skipped_too_long": [],
          "after_failed_edit_state": {
            "n": 4,
            "edit_calls": 4,
            "applied": 1,
            "apply_rate": 0.25,
            "not_found": 3,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 2,
            "other_tool": 0,
            "no_tool": 0,
            "loops": 0
          },
          "after_applied_edit_state": {
            "n": 12,
            "edit_calls": 11,
            "applied": 11,
            "apply_rate": 1.0,
            "not_found": 0,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 0,
            "other_tool": 1,
            "no_tool": 0,
            "loops": 0
          },
          "all": {
            "n": 16,
            "edit_calls": 15,
            "applied": 12,
            "apply_rate": 0.8,
            "not_found": 3,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 2,
            "other_tool": 1,
            "no_tool": 0,
            "loops": 0
          },
          "mean_tokens": 1127.3
        }
      },
      "edit": {
        "retry": {
          "n": 24,
          "same_text": 5,
          "applied": 1,
          "other_tool": 17,
          "no_tool": 0,
          "loops": 0,
          "mean_tokens": 711.4
        },
        "edit_states": {
          "skipped_too_long": [],
          "after_failed_edit_state": {
            "n": 4,
            "edit_calls": 4,
            "applied": 0,
            "apply_rate": 0.0,
            "not_found": 3,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 2,
            "other_tool": 0,
            "no_tool": 0,
            "loops": 0
          },
          "after_applied_edit_state": {
            "n": 12,
            "edit_calls": 10,
            "applied": 7,
            "apply_rate": 0.7,
            "not_found": 2,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 0,
            "other_tool": 2,
            "no_tool": 0,
            "loops": 0
          },
          "all": {
            "n": 16,
            "edit_calls": 14,
            "applied": 7,
            "apply_rate": 0.5,
            "not_found": 5,
            "identical": 0,
            "multiple": 0,
            "backslash_edits": 2,
            "other_tool": 2,
            "no_tool": 0,
            "loops": 0
          },
          "mean_tokens": 967.9
        }
      }
    }
  },
  "headline": "F16 same-text retries 5 to 5 of 24; apply on edit states 0.80 to 0.50",
  "system_monitor": {
    "samples": 187,
    "ram_free_gb": {
      "min": 9.08,
      "max": 19.35,
      "mean": 12.13
    },
    "gpu_shared_gb": {
      "min": 1.11,
      "max": 6.74,
      "mean": 5.03
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 90.45
    },
    "cpu_package_w": {
      "min": 15.3,
      "max": 40,
      "mean": 23.27
    },
    "cpu_core_w": {
      "min": 1,
      "max": 18.4,
      "mean": 3.48
    },
    "igpu_w": {
      "min": 0,
      "max": 25.3,
      "mean": 12.84
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 97,
      "mean": 12.55
    },
    "cpu_perf_pct": {
      "min": 29.6,
      "max": 159.6,
      "mean": 68.33
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
      "min": 1633,
      "max": 1918,
      "mean": 1816.42
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 0,
      "mean": 0.0
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

## Reading (2026-11-08 10:10 UTC)

Development split, one seed, the EXP-059 loop edit against the unedited model.
- 24 moments right after a failed edit: the next call resent the same old_string in 11 (unedited) and 5 (edit) at Q4_0, and in 5 and 5
  at F16; 0 and 1 retries applied at Q4_0, 1 and 1 at F16.
- 16 normal edit moments: edits applied 9 of 15 (60%) unedited and 7 of 13 (54%) with the edit at Q4_0; 12 of 15 (80%) and 7 of 14
  (50%) at F16, with "old_string not found" rising from 3 to 5.
The loop edit halves same-text retries at Q4_0 only, and at F16 it costs exact copying (80% to 50% of edits applied). With EXP-061
(code and MMLU costs), the edit is a trade-off, not a free improvement. One seed and 16 to 24 moments per cell: indicative only.

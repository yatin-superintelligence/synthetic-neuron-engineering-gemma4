# EXP-059: loop-edit-active-control

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-06T14:10:36.443695Z / 2026-11-06T17:18:24.058143Z
- **Wall time:** 187.8 min
- **Commit:** `0a1cc14b34356743f0782e65638e507cd0ea7581` (dirty=True)
- **Depends on:** EXP-053
- **Command:** `experiments/scripts/exp059_loop_edit_active_control.py`

## Why this experiment
EXP-053's control was far weaker than the edit, so its null result does not show that the edit is specific.

## Hypothesis (stated before running)
The loop edit lowers the loop rate at new seeds, and a matched-strength control does not.

## Configuration
```json
{
  "layers": [
    14,
    15,
    19,
    23
  ],
  "lam": -1.0,
  "seeds": [
    1,
    2
  ],
  "early_exit": true,
  "split": "dev",
  "control": "singular directions of matched gain, orthogonal to the loop direction, seed 59"
}
```

## Results
```json
{
  "arms": {
    "q4_0": {
      "base": {
        "n": 38,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2667,
          30
        ],
        "loop_rate_on_controls": [
          0.125,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7333,
          30
        ],
        "tool_call_rate_on_controls": [
          0.625,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          30
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 499.3
      },
      "remove_loop_dirs": {
        "n": 38,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0333,
          30
        ],
        "loop_rate_on_controls": [
          0.125,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.9667,
          30
        ],
        "tool_call_rate_on_controls": [
          0.625,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          30
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 462.9,
        "paired": {
          "pairs": 30,
          "lost": 7,
          "gained": 0,
          "mcnemar_p": 0.015625
        }
      },
      "remove_matched_dirs": {
        "n": 38,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3,
          30
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7,
          30
        ],
        "tool_call_rate_on_controls": [
          0.75,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          30
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 483.4,
        "paired": {
          "pairs": 30,
          "lost": 0,
          "gained": 1,
          "mcnemar_p": 1.0
        }
      }
    },
    "f16": {
      "base": {
        "n": 30,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3,
          30
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7,
          30
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          30
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 589.2
      },
      "remove_loop_dirs": {
        "n": 30,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0333,
          30
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.9667,
          30
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          30
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 395.7,
        "paired": {
          "pairs": 30,
          "lost": 9,
          "gained": 1,
          "mcnemar_p": 0.021484375
        }
      },
      "remove_matched_dirs": {
        "n": 30,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2333,
          30
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7667,
          30
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          30
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 494.9,
        "paired": {
          "pairs": 30,
          "lost": 3,
          "gained": 1,
          "mcnemar_p": 0.625
        }
      }
    }
  },
  "headline": "Q4_0 loops base 0.27, remove_loop_dirs 0.03, remove_matched_dirs 0.30; paired edit {'pairs': 30, 'lost': 7, 'gained': 0, 'mcnemar_p': 0.015625}, control {'pairs': 30, 'lost': 0, 'gained': 1, 'mcnemar_p': 1.0}",
  "system_monitor": {
    "samples": 158,
    "ram_free_gb": {
      "min": 9.65,
      "max": 20.48,
      "mean": 12.85
    },
    "gpu_shared_gb": {
      "min": 1.19,
      "max": 6.86,
      "mean": 5.01
    },
    "gpu_util_pct": {
      "min": 4,
      "max": 100,
      "mean": 91.94
    },
    "cpu_package_w": {
      "min": 18.4,
      "max": 35.5,
      "mean": 22.31
    },
    "cpu_core_w": {
      "min": 0.9,
      "max": 15.8,
      "mean": 2.29
    },
    "igpu_w": {
      "min": 0,
      "max": 21.2,
      "mean": 13.05
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 57,
      "mean": 9.38
    },
    "cpu_perf_pct": {
      "min": 26.9,
      "max": 165.8,
      "mean": 59.18
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
      "min": 1662,
      "max": 1918,
      "mean": 1836.53
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

## Reading
- **The effect replicated at new seeds at both precisions, and the matched control did not produce it.** See the table in the results above.
- **The p-values in `results.json` overstate the evidence.** They count 30 runs as independent. These come from 15 contexts in 10 tasks,
  and 3 tasks hold nearly all the effect, so the task-level sign-flip p is 0.25. Read the effect as large and consistent on these
  tasks, and not yet general. D-035 records this. EXP-066 tests held-out tasks.
- **The responses read as sensible.** In the 5 samples read at Q4_0, looping thoughts were replaced by task-relevant reasoning and a complete tool
  call, including a grep for the missing symbol. One blind edit and one case of remaining repetition show this is not a clean cure.
- **The control matches on gain but is still gentler:** at Q4_0 it left 9 of 30 response lengths unchanged, the edit none.

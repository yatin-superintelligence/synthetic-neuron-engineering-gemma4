# EXP-066: loop-edit-held-out

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-06T23:58:17.370637Z / 2026-11-07T09:06:34.731815Z
- **Wall time:** 548.3 min
- **Commit:** `11ba10345b0bccbc28f4f8bd4ec980e8b2901878` (dirty=True)
- **Depends on:** EXP-059, EXP-065
- **Command:** `experiments/scripts/exp066_loop_edit_held_out.py`

## Why this experiment
All evidence so far comes from 3 tasks of the dev split. A held-out test on 14 unseen tasks decides whether the effect is general.

## Hypothesis (stated before running)
The edit removes loops across several unseen tasks, and the control does not. If loops are removed in only 1 or 2 tasks, the effect is task-specific.

## Configuration
```json
{
  "split": "test",
  "seeds": [
    0,
    1
  ],
  "strength": 1.0,
  "layers": [
    14,
    15,
    19,
    23
  ],
  "primary": "task-level sign-flip",
  "early_exit": true
}
```

## Results
```json
{
  "arms": {
    "q4_0": {
      "base": {
        "n": 60,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3125,
          48
        ],
        "loop_rate_on_controls": [
          0.0,
          12
        ],
        "tool_call_rate_on_loop_contexts": [
          0.6875,
          48
        ],
        "tool_call_rate_on_controls": [
          0.75,
          12
        ],
        "forced_close_on_loop_contexts": [
          0.0208,
          48
        ],
        "forced_close_on_controls": [
          0.0,
          12
        ],
        "mean_tokens": 806.2
      },
      "remove_loop_dirs": {
        "n": 60,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0417,
          48
        ],
        "loop_rate_on_controls": [
          0.0,
          12
        ],
        "tool_call_rate_on_loop_contexts": [
          1.0,
          48
        ],
        "tool_call_rate_on_controls": [
          1.0,
          12
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          48
        ],
        "forced_close_on_controls": [
          0.0,
          12
        ],
        "mean_tokens": 661.6,
        "task_level": {
          "tasks": 14,
          "tasks_changed": 7,
          "net_loops_removed": 13,
          "sign_flip_p": 0.0143,
          "by_task": {
            "fastapi_11194": 1,
            "fastapi_14360": 1,
            "rich_3905": 2,
            "rich_4079": 2,
            "fastapi_12942": 2,
            "rich_2725": 1,
            "rich_3296": 4
          }
        }
      },
      "remove_matched_dirs": {
        "n": 60,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.25,
          48
        ],
        "loop_rate_on_controls": [
          0.0,
          12
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7292,
          48
        ],
        "tool_call_rate_on_controls": [
          0.9167,
          12
        ],
        "forced_close_on_loop_contexts": [
          0.0208,
          48
        ],
        "forced_close_on_controls": [
          0.0,
          12
        ],
        "mean_tokens": 936.5,
        "task_level": {
          "tasks": 14,
          "tasks_changed": 5,
          "net_loops_removed": 3,
          "sign_flip_p": 0.6885,
          "by_task": {
            "rich_3278": -1,
            "fastapi_12942": 1,
            "rich_2725": 1,
            "rich_3067": -1,
            "rich_3296": 3
          }
        }
      }
    },
    "f16": {
      "base": {
        "n": 48,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2917,
          48
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7083,
          48
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          48
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 790.7
      },
      "remove_loop_dirs": {
        "n": 48,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0,
          48
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          1.0,
          48
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          48
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 708.2,
        "task_level": {
          "tasks": 14,
          "tasks_changed": 7,
          "net_loops_removed": 14,
          "sign_flip_p": 0.0152,
          "by_task": {
            "fastapi_11194": 2,
            "rich_3278": 2,
            "rich_3905": 2,
            "rich_4079": 2,
            "fastapi_12942": 2,
            "rich_2725": 1,
            "rich_3296": 3
          }
        }
      },
      "remove_matched_dirs": {
        "n": 48,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3333,
          48
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.6667,
          48
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          48
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 908.6,
        "task_level": {
          "tasks": 14,
          "tasks_changed": 2,
          "net_loops_removed": -2,
          "sign_flip_p": 0.5037,
          "by_task": {
            "rich_2725": -1,
            "rich_3296": -1
          }
        }
      }
    }
  },
  "headline": "held out: Q4_0 loops base 0.31, edit 0.04, control 0.25, task p 0.0143; F16 0.29, 0.00, 0.33, task p 0.0152",
  "system_monitor": {
    "samples": 352,
    "ram_free_gb": {
      "min": 7.03,
      "max": 19.73,
      "mean": 11.97
    },
    "gpu_shared_gb": {
      "min": 1.16,
      "max": 6.88,
      "mean": 5.05
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 90.45
    },
    "cpu_package_w": {
      "min": 13.8,
      "max": 40.6,
      "mean": 21.65
    },
    "cpu_core_w": {
      "min": 0.6,
      "max": 20.4,
      "mean": 2.13
    },
    "igpu_w": {
      "min": 0,
      "max": 24.3,
      "mean": 12.71
    },
    "battery_pct": {
      "min": 43,
      "max": 100,
      "mean": 90.03
    },
    "cpu_pct": {
      "min": 0,
      "max": 99,
      "mean": 11.52
    },
    "cpu_perf_pct": {
      "min": 27.5,
      "max": 152,
      "mean": 60.36
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
      "min": 1508,
      "max": 1918,
      "mean": 1790.77
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 49373,
      "mean": 5550.75
    },
    "battery_charge_mw": {
      "min": 0,
      "max": 60006,
      "mean": 6058.01
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

Each replay restarts the model from a saved agent state, taken just before a response in which the unedited model looped in a real
run. There are 24 such moments from 14 tasks in the test split. None of these tasks was used to find the edit, set its strength or pick
its layers. Each moment ran at seeds 0 and 1, giving 48 replays per arm. A replay counts as a loop when one sentence repeats at least 8
times and the repetition score is at least 0.5.

| Of 48 replays | Q4_0 no edit | Q4_0 edit | Q4_0 control | F16 no edit | F16 edit | F16 control |
|---|---|---|---|---|---|---|
| Looped | 15 | 2 | 12 | 14 | 0 | 16 |
| Ended in a complete tool call | 33 | 48 | 35 | 34 | 48 | 32 |
| Mean tokens generated (Q4_0 over 60 replies including the 12 normal moments) | 806 | 662 | 937 | 791 | 708 | 909 |

Counted by task, with a sign-flip test across the 14 tasks (runs within a task are not independent):

| | Net loops removed | Tasks changed | Tasks made worse | p |
|---|---|---|---|---|
| Q4_0 edit | 13 | 7 | 0 | 0.0143 |
| Q4_0 control | 3 | 5 | 2 | 0.6885 |
| F16 edit | 14 | 7 | 0 | 0.0152 |
| F16 control | -2 | 2 | 2 | 0.5037 |

The control is a direction at right angles to the loop direction, mixed from the singular directions of the same weights whose gain is closest to it. Its gain is 78% to 100% of the edit's by layer (EXP-059 gains.json), so it is nearly, not exactly, as strong. It does nothing
measurable, so the effect belongs to the direction and not to any change of that size. Both precisions agree.

Normal moments (Q4_0 only, 12 replays at seed 0): 0 loops in every arm, and complete tool calls 9, 12 and 11 of 12.

Read from the saved texts (`analysis/read_texts.py`, counts only; the texts stay local because they hold task content):

- The edit changes what the model does next, not only whether it loops. At Q4_0 it called the same tool as the unedited model at the
  same moment and seed in 22 of 41 cases. It calls edit_file and submit_patch more often (17 and 6 of 60 replies, against 8 and 0).
  The control shifts the same way, less strongly (15 and 4), so part of this shift comes from any change at all.
- At F16, 4 of the edit's 48 tool calls have unbalanced string quotes, against 0 of 34 with no edit and 2 of 32 with the control.
  The official harness would return those as errors.
- Repeated runs of one to three words ("I will I will"): 2 in the F16 edit's 48 replies, 0 in every other arm. The first count of this
  read 0 everywhere because the pattern only matched single words. It was widened and checked against a known string before use.

What this shows: on unseen tasks, a rank-1 edit removes most thinking loops at both precisions, and a matched control does not. What it
does not show: that the edit raises tasks solved. On the official stack it did not (D-037, D-039). E4B, a model 1.5 times larger, had
no loops in its official run (D-041), which is the simple baseline this edit must be compared with.

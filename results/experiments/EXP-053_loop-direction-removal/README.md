# EXP-053: loop-direction-removal

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-06T11:45:08.369224Z / 2026-11-06T13:59:04.022673Z
- **Wall time:** 133.9 min
- **Commit:** `388d025b46066e142fced3fba774c0b0f5d13022` (dirty=True)
- **Depends on:** EXP-040
- **Command:** `experiments/scripts/exp053_loop_direction_removal.py`

## Why this experiment
EXP-040 is descriptive. Only an intervention can show that the direction carries looping, as Takehi et al. showed for Qwen. Loops took 31% of tokens in EXP-037 and preceded 10 of 15 context errors.

## Hypothesis (stated before running)
Removing the direction lowers the loop rate on loop contexts more than removing random directions, without lowering tool-call rate on controls.

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
    0
  ],
  "early_exit": true,
  "split": "dev",
  "control": "random unit directions in the same layers, seed 53"
}
```

## Results
```json
{
  "arms": {
    "q4_0": {
      "base": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3333,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.6667,
          15
        ],
        "tool_call_rate_on_controls": [
          0.875,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 495.1
      },
      "remove_loop_dirs": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.1333,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.8667,
          15
        ],
        "tool_call_rate_on_controls": [
          0.75,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 412.5
      },
      "remove_random_dirs": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3333,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.6667,
          15
        ],
        "tool_call_rate_on_controls": [
          0.875,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 476.5
      }
    },
    "f16": {
      "base": {
        "n": 15,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3333,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.6667,
          15
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 840.1
      },
      "remove_loop_dirs": {
        "n": 15,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          1.0,
          15
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 350.4
      },
      "remove_random_dirs": {
        "n": 15,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2667,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7333,
          15
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 591.3
      }
    }
  },
  "headline": "Q4_0 loop rate base 0.33, remove_loop_dirs 0.13, remove_random_dirs 0.33; control tool calls base 0.88, remove_loop_dirs 0.75, remove_random_dirs 0.88",
  "system_monitor": {
    "samples": 113,
    "ram_free_gb": {
      "min": 9.99,
      "max": 20.79,
      "mean": 13.18
    },
    "gpu_shared_gb": {
      "min": 1.22,
      "max": 6.78,
      "mean": 5.03
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 92.75
    },
    "cpu_package_w": {
      "min": 15.7,
      "max": 33.1,
      "mean": 21.73
    },
    "cpu_core_w": {
      "min": 0.7,
      "max": 12.7,
      "mean": 1.84
    },
    "igpu_w": {
      "min": 0,
      "max": 23.6,
      "mean": 13.38
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 99,
      "mean": 9.99
    },
    "cpu_perf_pct": {
      "min": 26,
      "max": 146.2,
      "mean": 49.15
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
      "min": 1704,
      "max": 1918,
      "mean": 1840.44
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
| Precision, 15 loop contexts | Base loops | Edit loops | Random loops | Base complete tool calls | Edit complete tool calls | Base tokens | Edit tokens |
|---|---|---|---|---|---|---|---|
| Q4_0 | 5 | 2 (lost 4, gained 1) | 5 (no change) | 10 | 13 | 9366 | 7814 |
| F16 | 5 | 0 (lost 5, gained 0) | 4 (lost 2, gained 1) | 10 | 15 | 12602 | 5256 |

- **Both precisions move the same way, and more strongly at F16**, the precision the direction was measured at. Base token counts stop
  early at confirmed loops, so the true saving is larger than shown.
- **Not yet evidence.** One seed. Paired tests give p = 0.375 at Q4_0 and p = 0.0625 at F16. The directions were fitted on half of
  these dev tasks, so the contexts are not independent of the edit.
- **The control is too weak for a clean comparison.** It changes much less: 6 of 15 loop responses kept their exact length, against
  1 of 15 for the edit. Still, at F16 even this tiny edit flipped 3 of 15 contexts, so a single loop label is fragile, and small
  counts here can move by chance. EXP-059 uses a matched control and new seeds.
- **Quality is unread.** A complete tool call is not a good one. EXP-053 did not save response text, so these calls cannot be judged.
  EXP-059 saves text. Loops are only worth removing if the calls that replace them are sensible.
- **Possible cost.** Control tool calls went from 7 to 6 of 8 at Q4_0, one state. F16 did not run controls in this experiment.
- rich_2943/9 at F16 kept a 3-gram repetition of 0.52 and 7 sentence repeats, under the loop threshold, so a loop can be weakened
  without being counted. The threshold was fixed before any edit, so the count stands, but it is not the whole picture.

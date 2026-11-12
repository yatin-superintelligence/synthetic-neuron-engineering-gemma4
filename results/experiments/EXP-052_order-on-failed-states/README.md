# EXP-052: order-on-failed-states

- **Status:** OK
- **Category:** localization
- **Started / finished (UTC):** 2026-11-08T15:21:33.906918Z / 2026-11-08T16:24:54.385348Z
- **Wall time:** 63.3 min
- **Commit:** `6d3e5dd35f6f0e83848aa4efab09300169f290ad` (dirty=True)
- **Depends on:** EXP-041
- **Command:** `experiments/scripts/exp052_order_on_failed_states.py`

## Why this experiment
EXP-041 was short of failed states, and the observed 9 of 10 against 91 of 207 may be selection. This decides whether the order neuron (EXP-042) is worth building.

## Hypothesis (stated before running)
Writing old_string first turns failed edits into applied ones more often than the new-first branch. If the gain CI includes zero, H-4's order cause is dropped.

## Configuration
```json
{
  "split": "dev",
  "seeds": [
    0,
    1
  ],
  "max_prompt_tokens": 16000,
  "kinds": [
    "mostly_real_altered",
    "identical",
    "invented",
    "lines_real_block_not",
    "over_escaped"
  ],
  "slots": 1,
  "per_run_task": 4
}
```

## Results
```json
{
  "order": {
    "q4_0": {
      "pairs": 42,
      "tasks": 9,
      "rate_old_first": 0.1429,
      "rate_new_first": 0.0476,
      "seed0_only_old_first": 3,
      "seed0_only_new_first": 1,
      "mcnemar_p_seed0": 0.625,
      "mean_gain": 0.0952,
      "task_bootstrap_ci": [
        0.0,
        0.3077
      ],
      "outcomes_old_first": {
        "not_found": 34,
        "applied": 6,
        "multiple": 2
      },
      "outcomes_new_first": {
        "not_found": 35,
        "identical": 5,
        "applied": 2
      }
    },
    "f16": {
      "pairs": 42,
      "tasks": 9,
      "rate_old_first": 0.119,
      "rate_new_first": 0.0,
      "seed0_only_old_first": 2,
      "seed0_only_new_first": 0,
      "mcnemar_p_seed0": 0.5,
      "mean_gain": 0.119,
      "task_bootstrap_ci": [
        0.0,
        0.3235
      ],
      "outcomes_old_first": {
        "not_found": 34,
        "applied": 5,
        "multiple": 2,
        "identical": 1
      },
      "outcomes_new_first": {
        "not_found": 32,
        "identical": 7,
        "file_missing": 2,
        "empty": 1
      }
    }
  },
  "headline": "failed states, forced order: Q4_0 0.14 old first vs 0.05 new first on 42 pairs, gain CI [0.0, 0.3077]; F16 0.12 vs 0.00, gain CI [0.0, 0.3235]",
  "system_monitor": {
    "samples": 53,
    "ram_free_gb": {
      "min": 8.88,
      "max": 19.69,
      "mean": 12.02
    },
    "gpu_shared_gb": {
      "min": 1.35,
      "max": 6.94,
      "mean": 5.25
    },
    "gpu_util_pct": {
      "min": 3,
      "max": 100,
      "mean": 88.57
    },
    "cpu_package_w": {
      "min": 15.6,
      "max": 38.4,
      "mean": 24.48
    },
    "cpu_core_w": {
      "min": 1.3,
      "max": 17.8,
      "mean": 4.86
    },
    "igpu_w": {
      "min": 0,
      "max": 22.3,
      "mean": 12.45
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 69,
      "mean": 12.42
    },
    "cpu_perf_pct": {
      "min": 45.8,
      "max": 138.1,
      "mean": 81.99
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
      "mean": 1771.09
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

# EXP-041: edit-failure-replay

- **Status:** OK
- **Category:** localization
- **Started / finished (UTC):** 2026-11-06T07:13:13.173016Z / 2026-11-06T08:24:25.793801Z
- **Wall time:** 71.2 min
- **Commit:** `3b00bcea401caa340bd4f0061b88af71a9603676` (dirty=True)
- **Depends on:** EXP-036
- **Command:** `experiments/scripts/exp041_edit_failure_replay.py`

## Why this experiment
50% of edit_file calls failed in EXP-036, 11 of 58 tasks never applied an edit, and none of those resolved. Reading the traces suggests two structural causes that a weight edit could address.

## Hypothesis (stated before running)
Forcing old_string first raises the apply rate over the new_string-first branch.

## Configuration
```json
{
  "traces": [
    "exp036",
    "exp037"
  ],
  "split": "dev",
  "order_seeds": [
    0,
    1
  ],
  "max_prompt_tokens": 9000,
  "slots": 6,
  "per_run_task": 4
}
```

## Results
```json
{
  "census": {
    "exp036": {
      "edits": 86,
      "applied": 43,
      "failed": 43,
      "kinds": {
        "applied": 43,
        "mostly_real_altered": 11,
        "over_escaped": 9,
        "identical": 8,
        "invented": 5,
        "stale_after_own_edit": 4,
        "multiple": 4,
        "lines_real_block_not": 2
      },
      "failed_repeating_identical_call": 12,
      "failed_with_backslash": 19
    },
    "exp037": {
      "edits": 132,
      "applied": 58,
      "failed": 74,
      "kinds": {
        "applied": 58,
        "invented": 29,
        "mostly_real_altered": 15,
        "multiple": 15,
        "stale_after_own_edit": 12,
        "identical": 2,
        "lines_real_block_not": 1
      },
      "failed_repeating_identical_call": 33,
      "failed_with_backslash": 32
    }
  },
  "replay": {
    "q4_0": {
      "argument_order": {
        "pairs": 62,
        "tasks": 17,
        "rate_old_first": 0.8226,
        "rate_new_first": 0.7903,
        "seed0_only_old_first": 3,
        "seed0_only_new_first": 2,
        "mcnemar_p_seed0": 1.0,
        "mean_gain": 0.0323,
        "task_bootstrap_ci": [
          -0.0833,
          0.1282
        ],
        "outcomes_old_first": {
          "applied": 51,
          "multiple": 5,
          "not_found": 6
        },
        "outcomes_new_first": {
          "applied": 49,
          "not_found": 7,
          "identical": 2,
          "multiple": 2,
          "file_missing": 1,
          "empty": 1
        }
      }
    },
    "f16": {
      "argument_order": {
        "pairs": 62,
        "tasks": 17,
        "rate_old_first": 0.7903,
        "rate_new_first": 0.8387,
        "seed0_only_old_first": 1,
        "seed0_only_new_first": 3,
        "mcnemar_p_seed0": 0.625,
        "mean_gain": -0.0484,
        "task_bootstrap_ci": [
          -0.1714,
          0.0556
        ],
        "outcomes_old_first": {
          "applied": 49,
          "multiple": 6,
          "file_missing": 1,
          "not_found": 6
        },
        "outcomes_new_first": {
          "applied": 52,
          "not_found": 6,
          "identical": 2,
          "multiple": 2
        }
      }
    }
  },
  "headline": "forced order: Q4_0 0.82 old first vs 0.79 new first on 62 pairs, gain CI [-0.0833, 0.1282]; F16 0.79 vs 0.84, gain CI [-0.1714, 0.0556]",
  "system_monitor": {
    "samples": 63,
    "ram_free_gb": {
      "min": 7.07,
      "max": 19.19,
      "mean": 10.61
    },
    "gpu_shared_gb": {
      "min": 1.24,
      "max": 7.98,
      "mean": 6.02
    },
    "gpu_util_pct": {
      "min": 0.3,
      "max": 125.7,
      "mean": 85.29
    },
    "thermal_c": {
      "min": 27.9,
      "max": 27.9,
      "mean": 27.9
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 29.8,
      "mean": 9.44
    },
    "cpu_perf_pct": {
      "min": 33.5,
      "max": 171.9,
      "mean": 80.35
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_freq_mhz": {
      "min": 1643,
      "max": 1918,
      "mean": 1815.48
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
- **No order effect on these states.** Q4_0: 82% old first, 79% new first, gain CI -8% to +13%. F16: 79% against 84%, gain CI
  -17% to +6%. The two precisions disagree in sign, and both intervals cover zero.
- **Weak test of the claim that matters.** The 9000-token limit left 31 states, only 5 of them failed edits, because failed edits
  come later in a task. Applied states apply either way (Q4_0 49/52 in both branches; F16 47/52 and 52/52), so they cannot show
  a gain. Failed states applied 2/10 old first and 0/10 new first at both precisions, from one state.
- The earlier 9 of 10 against 91 of 207 is therefore not supported as a cause on these states. It may be selection: the model
  may write old_string first when it is already sure of the text. EXP-052 tests failed states directly.
- Preflight for this run tested 0 states and still passed, through a race with a code change. The real run was checked by
  hand, and experiments now raise an error on an empty state set.

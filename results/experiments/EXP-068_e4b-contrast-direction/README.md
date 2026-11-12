# EXP-068: e4b-contrast-direction

- **Status:** OK
- **Category:** localization
- **Started / finished (UTC):** 2026-11-08T05:53:03.739424Z / 2026-11-08T06:11:41.996884Z
- **Wall time:** 18.6 min
- **Commit:** `315e01d918a0a8838df8470d28a16e2fab86f213` (dirty=True)
- **Depends on:** EXP-040, EXP-071
- **Command:** `experiments/scripts/exp068_e4b_contrast_direction.py`

## Why this experiment
E4B leaves the states that trap E2B. Contrasting E2B's state on its own loop with its state on E4B's continuation of the same prompt isolates looping from lateness and context.

## Hypothesis (stated before running)
v_T agrees with EXP-040's direction (cosine at least 0.9 at the edit layers). If not, it separates held-out loop from teacher tokens better and becomes a new edit.

## Configuration
```json
{
  "ctx_tokens": 3000,
  "resp_tokens": 1500,
  "tokens_per_class": 200,
  "split": "dev",
  "fit_half_seed": 40,
  "classes": [
    "LOOP",
    "PRE",
    "TEACHER"
  ],
  "edit_layers": [
    14,
    15,
    19,
    23
  ],
  "cosine_rule": 0.9
}
```

## Results
```json
{
  "edit_layers": {
    "14": {
      "layer": 14,
      "d_loop_vs_teacher_heldout": 6.419,
      "random_abs_d_max": 2.763,
      "old_direction_d_loop_vs_teacher": 6.311,
      "new_direction_d_loop_vs_pre": 4.582,
      "cosine_with_exp040": 0.988,
      "n_tokens": {
        "fit_loop": 1200,
        "fit_teacher": 3984,
        "hold_loop": 1800,
        "hold_teacher": 4710,
        "hold_pre": 1766
      }
    },
    "15": {
      "layer": 15,
      "d_loop_vs_teacher_heldout": 4.733,
      "random_abs_d_max": 1.581,
      "old_direction_d_loop_vs_teacher": 4.645,
      "new_direction_d_loop_vs_pre": 3.684,
      "cosine_with_exp040": 0.989,
      "n_tokens": {
        "fit_loop": 1200,
        "fit_teacher": 3984,
        "hold_loop": 1800,
        "hold_teacher": 4710,
        "hold_pre": 1766
      }
    },
    "19": {
      "layer": 19,
      "d_loop_vs_teacher_heldout": 4.043,
      "random_abs_d_max": 1.968,
      "old_direction_d_loop_vs_teacher": 3.958,
      "new_direction_d_loop_vs_pre": 3.594,
      "cosine_with_exp040": 0.988,
      "n_tokens": {
        "fit_loop": 1200,
        "fit_teacher": 3984,
        "hold_loop": 1800,
        "hold_teacher": 4710,
        "hold_pre": 1766
      }
    },
    "23": {
      "layer": 23,
      "d_loop_vs_teacher_heldout": 3.95,
      "random_abs_d_max": 2.12,
      "old_direction_d_loop_vs_teacher": 3.899,
      "new_direction_d_loop_vs_pre": 3.324,
      "cosine_with_exp040": 0.994,
      "n_tokens": {
        "fit_loop": 1200,
        "fit_teacher": 3984,
        "hold_loop": 1800,
        "hold_teacher": 4710,
        "hold_pre": 1766
      }
    }
  },
  "decision": "same axis as EXP-040",
  "headline": "same axis as EXP-040; cosine with EXP-040 at layers 14/15/19/23: 0.988, 0.989, 0.988, 0.994; held-out d' loop vs E4B: new 6.419 old 6.311, new 4.733 old 4.645, new 4.043 old 3.958, new 3.95 old 3.899",
  "system_monitor": {
    "samples": 16,
    "ram_free_gb": {
      "min": 3.42,
      "max": 19.06,
      "mean": 5.19
    },
    "gpu_shared_gb": {
      "min": 1.18,
      "max": 8.39,
      "mean": 7.88
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 82.06
    },
    "cpu_package_w": {
      "min": 17.1,
      "max": 55,
      "mean": 39.46
    },
    "cpu_core_w": {
      "min": 7,
      "max": 22,
      "mean": 11.12
    },
    "igpu_w": {
      "min": 0,
      "max": 27,
      "mean": 21.86
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 24.8,
      "mean": 13.13
    },
    "cpu_perf_pct": {
      "min": 120.3,
      "max": 176.2,
      "mean": 145.64
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
      "min": 1746,
      "max": 1867,
      "mean": 1815.62
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

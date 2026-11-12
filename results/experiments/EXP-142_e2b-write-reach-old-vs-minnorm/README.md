# EXP-142: e2b-write-reach-old-vs-minnorm

- **Status:** OK
- **Category:** diagnosis
- **Started / finished (UTC):** 2026-11-11T10:51:23.047213Z / 2026-11-11T11:48:07.906160Z
- **Wall time:** 56.7 min
- **Commit:** `2612ff3715f747e479f516adc7b9ddefcf042d53` (dirty=True)
- **Depends on:** EXP-117, EXP-118, EXP-140b
- **Command:** `experiments/scripts/exp142_e2b_write_reach.py`

## Why this experiment
The paper's claim that a last-layer neuron cannot write the sentence's first word was measured with the old write direction, which D-170 found about ten times longer than needed for this token.

## Hypothesis (stated before running)
With the minimum-norm direction the first word becomes the most probable at most of the states at some strength; with the old direction it does not (EXP-118: 5.1 logits behind at 32 times).

## Configuration
```json
{
  "data": "artifacts_local/EXP-117/thought_starts_dev.jsonl (after read_file)",
  "states": 24,
  "scales_of_y_norm": [
    0.25,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    256,
    1000
  ],
  "layer": 34,
  "max_tokens": 3000,
  "note": "the last 3,000 prompt tokens"
}
```

## Results
```json
{
  "reach": {
    "states": 24,
    "competitors": [
      [
        818,
        "The"
      ],
      [
        236777,
        "I"
      ],
      [
        95889,
        "Neither"
      ]
    ],
    "scales": [
      0.25,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      256,
      1000
    ],
    "unedited_gap_median": 11.062,
    "old": {
      "states_where_Before_becomes_top": 0,
      "best_margin_median": -2.75,
      "best_margin_min": -6.0,
      "margin_median_by_scale": {
        "0.25": -2.75,
        "1": -6.38,
        "2": -5.44,
        "4": -5.19,
        "8": -5.25,
        "16": -5.19,
        "32": -5.25,
        "64": -5.25,
        "256": -5.22,
        "1000": -5.25
      }
    },
    "minimum_norm": {
      "states_where_Before_becomes_top": 24,
      "best_margin_median": 1.94,
      "best_margin_min": 0.88,
      "margin_median_by_scale": {
        "0.25": 1.94,
        "1": 0.12,
        "2": 0.0,
        "4": 0.0,
        "8": 0.0,
        "16": 0.0,
        "32": 0.0,
        "64": 0.0,
        "256": 0.0,
        "1000": 0.0
      }
    }
  },
  "headline": "E2B layer-34 write, 'Before' at 24 thought starts after a read: old direction top at 0 (best margin median -2.75), minimum-norm at 24 (best margin median 1.94)",
  "system_monitor": {
    "samples": 49,
    "ram_free_gb": {
      "min": 7.1,
      "max": 20.3,
      "mean": 7.5
    },
    "gpu_shared_gb": {
      "min": 1.02,
      "max": 6.99,
      "mean": 6.81
    },
    "gpu_util_pct": {
      "min": 5,
      "max": 100,
      "mean": 95.47
    },
    "cpu_package_w": {
      "min": 15.9,
      "max": 52.7,
      "mean": 42.33
    },
    "cpu_core_w": {
      "min": 5.8,
      "max": 17.9,
      "mean": 9.42
    },
    "igpu_w": {
      "min": 0,
      "max": 28,
      "mean": 25.94
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 96,
      "mean": 10.93
    },
    "cpu_perf_pct": {
      "min": 117.3,
      "max": 177.5,
      "mean": 151.36
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
      "max": 1900,
      "mean": 1837.63
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

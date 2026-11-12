# EXP-004: determinism-and-batching

- **Status:** OK
- **Category:** infrastructure
- **Started / finished (UTC):** 2026-11-06T00:36:44.950905Z / 2026-11-06T03:08:48.655034Z
- **Wall time:** 152.1 min
- **Commit:** `b609a4ba99131cd9302c2ef4ed2ce1d1c2f46eed` (dirty=True)
- **Depends on:** EXP-001
- **Command:** `experiments/scripts/exp004_determinism.py`

## Why this experiment
Quantifies the evaluation noise floor that every edit comparison must exceed.

## Hypothesis (stated before running)
Run-to-run outputs at fixed parallelism are mostly identical; changing parallelism alters a few percent of generations but pass-rate changes stay within +-1-2 points.

## Configuration
```json
{
  "suite": "mbpp_dev",
  "subset_parallel1": 60
}
```

## Results
```json
{
  "f16": {
    "identical_run1_run2": 179,
    "n": 189,
    "identical_p8_vs_p1": 47,
    "n_p1": 60,
    "pass_run1": 0.6031746031746031,
    "pass_run2": 0.6031746031746031,
    "flips_run1_run2": 0,
    "pass_p8_subset": 0.65,
    "pass_p1_subset": 0.6666666666666666,
    "flips_p8_p1": 1
  },
  "q4_0": {
    "identical_run1_run2": 87,
    "n": 189,
    "identical_p8_vs_p1": 33,
    "n_p1": 60,
    "pass_run1": 0.6296296296296297,
    "pass_run2": 0.6349206349206349,
    "flips_run1_run2": 9,
    "pass_p8_subset": 0.7166666666666667,
    "pass_p1_subset": 0.7333333333333333,
    "flips_p8_p1": 1
  },
  "headline": "f16: run-run identical 179/189, p8-vs-p1 identical 47/60, outcome flips 0; q4_0: run-run identical 87/189, p8-vs-p1 identical 33/60, outcome flips 9",
  "system_monitor": {
    "samples": 132,
    "ram_free_gb": {
      "min": 8.54,
      "max": 20.02,
      "mean": 11.5
    },
    "gpu_shared_gb": {
      "min": 1.37,
      "max": 7.56,
      "mean": 5.62
    },
    "gpu_util_pct": {
      "min": 0.2,
      "max": 98.2,
      "mean": 88.25
    },
    "thermal_c": {
      "min": 27.9,
      "max": 27.9,
      "mean": 27.9
    },
    "battery_pct": {
      "min": 76,
      "max": 100,
      "mean": 97.44
    },
    "cpu_pct": {
      "min": 0,
      "max": 25.5,
      "mean": 9.45
    },
    "cpu_perf_pct": {
      "min": 39,
      "max": 166.2,
      "mean": 62.98
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_freq_mhz": {
      "min": 1489,
      "max": 1918,
      "mean": 1778.77
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 36192,
      "mean": 6161.31
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

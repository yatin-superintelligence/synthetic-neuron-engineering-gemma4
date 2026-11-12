# EXP-028: official-harness-controls-local

- **Status:** OK
- **Category:** benchmark
- **Started / finished (UTC):** 2026-11-06T05:43:24.884567Z / 2026-11-06T05:51:12.055715Z
- **Wall time:** 7.8 min
- **Commit:** `c17af000d8cd21ac7b93e0335fc0f6cd33877b1e` (dirty=True)
- **Depends on:** EXP-002
- **Command:** `experiments/scripts/exp028_official_controls.py`

## Why this experiment
Every local real-task score depends on the official harness behaving here as on Kaggle. Reference fixes must pass and empty patches must fail, per task.

## Hypothesis (stated before running)
Most fastapi and rich small-fix tasks pass both controls. requests tasks cannot pass in this path: its src/ layout is not importable in subprocess mode, so they fail the gold control.

## Configuration
```json
{
  "harness": "swegemma 0.2.11",
  "adk": "1.36.1",
  "workers": 4,
  "sandbox": "subprocess inside the sne-harness Linux container, as on Kaggle"
}
```

## Results
```json
{
  "n": 58,
  "sound": 45,
  "dropped": [
    "requests_6589",
    "requests_6592",
    "requests_6629",
    "requests_6644",
    "requests_7205",
    "requests_7315",
    "requests_7328",
    "requests_7427",
    "requests_7433",
    "requests_7502",
    "rich_3296",
    "rich_3469",
    "rich_4079"
  ],
  "by_repo": {
    "requests": [
      0,
      10
    ],
    "fastapi": [
      24,
      24
    ],
    "rich": [
      21,
      24
    ]
  },
  "agrees_with_public_audit": 49,
  "headline": "45/58 sound {'requests': [0, 10], 'fastapi': [24, 24], 'rich': [21, 24]}; agrees with public audit on 49/58",
  "system_monitor": {
    "samples": 7,
    "ram_free_gb": {
      "min": 4.39,
      "max": 16.71,
      "mean": 9.71
    },
    "gpu_shared_gb": {
      "min": 1.16,
      "max": 1.29,
      "mean": 1.23
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 6.7,
      "mean": 1.44
    },
    "thermal_c": {
      "min": 27.9,
      "max": 27.9,
      "mean": 27.9
    },
    "battery_pct": {
      "min": 69,
      "max": 76,
      "mean": 72.14
    },
    "cpu_pct": {
      "min": 9.6,
      "max": 22.3,
      "mean": 14.46
    },
    "cpu_perf_pct": {
      "min": 132.9,
      "max": 176.5,
      "mean": 155.41
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_freq_mhz": {
      "min": 1797,
      "max": 1876,
      "mean": 1837.14
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 0,
      "mean": 0.0
    },
    "battery_charge_mw": {
      "min": 27195,
      "max": 55146,
      "mean": 48317.29
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

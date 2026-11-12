# EXP-124: local-transplant-correctness

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-11T06:34:42.908500Z / 2026-11-11T10:30:48.573827Z
- **Wall time:** 236.1 min
- **Commit:** `138d946ac116b6d43177c91460c1b062d6f5e26b` (dirty=True)
- **Depends on:** EXP-102, EXP-112, EXP-116
- **Command:** `experiments/scripts/exp124_local_transplant_correctness.py`

## Why this experiment
E2B fails mainly at writing the correct fix; the E4B transplant is the one edit with a replicated capability gain, and its effect on correct first edits at real states was never measured.

## Hypothesis (stated before running)
On FEB states whose file the reference touches, the transplant raises first edits that leave the reference change above unedited (exact McNemar).

## Configuration
```json
{
  "adapter": "EXP-102/st_a1",
  "precision": "f16",
  "max_prompt": 12000,
  "sampling": {
    "temperature": 0.2,
    "top_p": 0.95
  }
}
```

## Results
```json
{
  "states": 233,
  "summary": {
    "base": {
      "counts": {
        "applies": 107,
        "other_tool": 89,
        "not_found": 26,
        "empty": 5,
        "noop": 4,
        "no_call": 1,
        "multiple": 1
      },
      "applies": 107,
      "holds_reference": 26
    },
    "e4b_tx": {
      "counts": {
        "applies": 118,
        "other_tool": 78,
        "not_found": 25,
        "noop": 4,
        "multiple": 3,
        "no_call": 4,
        "empty": 1
      },
      "applies": 118,
      "holds_reference": 22
    }
  },
  "tests": {
    "applies": {
      "transplant_only": 39,
      "base_only": 28,
      "p": 0.2215
    },
    "holds_reference": {
      "transplant_only": 8,
      "base_only": 12,
      "p": 0.5034
    }
  },
  "headline": "F16 FEB, 233 states whose file the reference touches: applies base 107, transplant 118; reference change held base 26, transplant 22",
  "system_monitor": {
    "samples": 842,
    "ram_free_gb": {
      "min": 0,
      "max": 21.55,
      "mean": 9.99
    },
    "gpu_shared_gb": {
      "min": 0.98,
      "max": 7.22,
      "mean": 6.6
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 91.58
    },
    "cpu_package_w": {
      "min": 15.5,
      "max": 68.9,
      "mean": 25.67
    },
    "cpu_core_w": {
      "min": 1,
      "max": 47.5,
      "mean": 4.42
    },
    "igpu_w": {
      "min": 0,
      "max": 25.1,
      "mean": 13.23
    },
    "battery_pct": {
      "min": 80,
      "max": 100,
      "mean": 97.79
    },
    "cpu_pct": {
      "min": 0,
      "max": 100,
      "mean": 13.59
    },
    "cpu_perf_pct": {
      "min": 27,
      "max": 159.6,
      "mean": 75.59
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
      "mean": 1804.84
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 31904,
      "mean": 116.73
    },
    "battery_charge_mw": {
      "min": 0,
      "max": 44099,
      "mean": 1002.28
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

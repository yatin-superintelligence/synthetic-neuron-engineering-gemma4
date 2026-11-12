# EXP-003: engine-fidelity-hf-vs-llamacpp

- **Status:** OK
- **Category:** infrastructure
- **Started / finished (UTC):** 2026-11-05T22:04:24.388879Z / 2026-11-05T22:09:19.000199Z
- **Wall time:** 4.8 min
- **Commit:** `2f759a29587e5651bfb498c76d02d4dc118b53e8` (dirty=True)
- **Depends on:** EXP-001
- **Command:** `experiments/scripts/exp003_engine_fidelity.py`

## Why this experiment
The fast engine must compute the same function as the reference implementation, otherwise edits derived in PyTorch would not transfer.

## Hypothesis (stated before running)
HF agrees with llama.cpp F16 on over 98% of teacher-forced next tokens, and slightly less with Q4_0 continuations.

## Configuration
```json
{
  "n_prompts": 40,
  "new_tokens": 128,
  "source": "mbpp_full_train first 40 by file order",
  "method": "teacher-forced argmax agreement, HF bf16 batch size 1"
}
```

## Results
```json
{
  "f16": {
    "tokens": 5083,
    "argmax_agreement": 0.99548,
    "mean_prob_of_token": 0.9216,
    "sequences_fully_agreeing": 22,
    "control_shuffled_agreement": 0.02538,
    "seconds": 75.0
  },
  "q4_0": {
    "tokens": 5038,
    "argmax_agreement": 0.97737,
    "mean_prob_of_token": 0.9167,
    "sequences_fully_agreeing": 1,
    "control_shuffled_agreement": 0.0268,
    "seconds": 27.6
  },
  "headline": "HF argmax agreement: F16 0.9955, Q4_0 0.9774; shuffled control 0.025",
  "system_monitor": {
    "samples": 5,
    "ram_free_gb": {
      "min": 9.96,
      "max": 22.33,
      "mean": 14.19
    },
    "gpu_shared_gb": {
      "min": 1.37,
      "max": 7.55,
      "mean": 5.36
    },
    "gpu_util_pct": {
      "min": 0.4,
      "max": 91.2,
      "mean": 37.02
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
      "max": 13.7,
      "mean": 6.2
    },
    "cpu_perf_pct": {
      "min": 48.3,
      "max": 155.5,
      "mean": 107.68
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_freq_mhz": {
      "min": 1787,
      "max": 1900,
      "mean": 1843.4
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

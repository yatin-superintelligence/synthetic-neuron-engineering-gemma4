# EXP-162: 31b-causal-dose

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-12T15:03:13.301491Z / 2026-11-12T15:13:03.796149Z
- **Wall time:** 9.8 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-160, EXP-161a
- **Command:** `experiments/scripts/exp162_31b_causal_dose.py`

## Why this experiment
EXP-160's causal neurons moved 31B a quarter of the way to the passing fix on 7 pairs and did not raise tasks solved; the open questions are whether the effect holds on all pairs and whether a stronger edit closes the gap.

## Hypothesis (stated before running)
On all pairs, the K=32 x2 edit chosen on the other fold raises the held-out preference in at least 65% of pairs and more than random neurons. Prediction for the dose: no configuration with damage of 0.10 nats per token or less flips the preference in a quarter more of the held-out pairs than are already flipped unedited.

## Configuration
```json
{
  "model": "google/gemma-4-31B-it-qat-q4_0-unquantized",
  "bench": "artifacts_local/EXP-147/success_bench.jsonl.gz",
  "max_call_tokens": 64,
  "pairs_per_fail": 1,
  "own_window": 128,
  "prefill_chunk": 1024,
  "truncate_to": [
    1024,
    7168
  ],
  "configs": [
    [
      "chosen",
      32,
      1.0
    ],
    [
      "chosen",
      32,
      3.0
    ],
    [
      "chosen",
      128,
      1.0
    ],
    [
      "chosen",
      128,
      3.0
    ],
    [
      "random",
      32,
      1.0
    ],
    [
      "random",
      128,
      3.0
    ]
  ]
}
```

## Results
```json
{
  "headline": "6 pairs on 5 tasks; held out, chosen K=32 x2 raises the preference in 6 of 6 (random 2); dose leads: none",
  "fit": {
    "pairs": 6,
    "tasks": 5,
    "truncated": 3,
    "skipped": 32,
    "preference_mean": -0.6149,
    "held_out_pairs": 6,
    "summary": {
      "chosen_32_x2": {
        "pairs": 6,
        "raises": 6,
        "mean_change": 0.2371,
        "flips": 0,
        "base_flips": 0,
        "mean_damage": 0.0112,
        "mean_gap": -0.6149
      },
      "chosen_32_x4": {
        "pairs": 6,
        "raises": 6,
        "mean_change": 0.3244,
        "flips": 0,
        "base_flips": 0,
        "mean_damage": 0.1771,
        "mean_gap": -0.6149
      },
      "chosen_128_x2": {
        "pairs": 6,
        "raises": 6,
        "mean_change": 0.3603,
        "flips": 0,
        "base_flips": 0,
        "mean_damage": 0.0678,
        "mean_gap": -0.6149
      },
      "chosen_128_x4": {
        "pairs": 6,
        "raises": 6,
        "mean_change": 0.3974,
        "flips": 1,
        "base_flips": 0,
        "mean_damage": 0.947,
        "mean_gap": -0.6149
      },
      "random_32_x2": {
        "pairs": 6,
        "raises": 2,
        "mean_change": 0.0007,
        "flips": 0,
        "base_flips": 0,
        "mean_damage": 0.0056,
        "mean_gap": -0.6149
      },
      "random_128_x4": {
        "pairs": 6,
        "raises": 2,
        "mean_change": 0.0005,
        "flips": 0,
        "base_flips": 0,
        "mean_damage": 0.0049,
        "mean_gap": -0.6149
      }
    },
    "replication_passed": true,
    "dose_leads": [],
    "saved_config": "chosen_128_x4"
  },
  "system_monitor": {
    "samples": 10
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

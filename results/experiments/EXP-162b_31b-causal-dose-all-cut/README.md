# EXP-162b: 31b-causal-dose-all-cut

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-12T15:17:52.654663Z / 2026-11-12T15:26:42.624180Z
- **Wall time:** 8.8 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-160, EXP-161a
- **Command:** `/content/repo/experiments/scripts/exp162_31b_causal_dose.py`

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
    4096
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
  "headline": "25 pairs on 12 tasks; held out, chosen K=32 x2 raises the preference in 20 of 25 (random 8); dose leads: none",
  "fit": {
    "pairs": 25,
    "tasks": 12,
    "truncated": 25,
    "skipped": 13,
    "preference_mean": -0.4565,
    "held_out_pairs": 25,
    "summary": {
      "chosen_32_x2": {
        "pairs": 25,
        "raises": 20,
        "mean_change": 0.2007,
        "flips": 4,
        "base_flips": 3,
        "mean_damage": 0.0502,
        "mean_gap": -0.4565
      },
      "chosen_32_x4": {
        "pairs": 25,
        "raises": 20,
        "mean_change": 0.2482,
        "flips": 5,
        "base_flips": 3,
        "mean_damage": 0.3383,
        "mean_gap": -0.4565
      },
      "chosen_128_x2": {
        "pairs": 25,
        "raises": 20,
        "mean_change": 0.2599,
        "flips": 6,
        "base_flips": 3,
        "mean_damage": 0.0956,
        "mean_gap": -0.4565
      },
      "chosen_128_x4": {
        "pairs": 25,
        "raises": 20,
        "mean_change": 0.3245,
        "flips": 7,
        "base_flips": 3,
        "mean_damage": 0.7943,
        "mean_gap": -0.4565
      },
      "random_32_x2": {
        "pairs": 25,
        "raises": 8,
        "mean_change": -0.001,
        "flips": 3,
        "base_flips": 3,
        "mean_damage": -0.0025,
        "mean_gap": -0.4565
      },
      "random_128_x4": {
        "pairs": 25,
        "raises": 14,
        "mean_change": 0.0038,
        "flips": 3,
        "base_flips": 3,
        "mean_damage": -0.0022,
        "mean_gap": -0.4565
      }
    },
    "replication_passed": true,
    "dose_leads": [],
    "saved_config": "chosen_128_x4"
  },
  "system_monitor": {
    "samples": 9
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

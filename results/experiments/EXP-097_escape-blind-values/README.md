# EXP-097: escape-blind-values

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T23:11:08.473172Z / 2026-11-09T01:38:02.890158Z
- **Wall time:** 146.9 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-092, EXP-096, EXP-094b
- **Command:** `experiments/scripts/exp097_escape_blind_values.py`

## Why this experiment
E2B copies the JSON escape of the harness's file view into its edits; output-side writes are capped by Gemma 4's norms (EXP-096), so this edit removes the escape feature from what the attention heads read, as an exact rank-1 v_proj LoRA.

## Hypothesis (stated before running)
Escape-blind values raise the share of applying bench answers on held-out states against the unedited model in the same server (exact McNemar p < 0.05) and against a random-direction control, with no fewer correct answers and no significant MBPP+ loss.

## Configuration
```json
{
  "layer_sets": {
    "L13-14": [
      13,
      14
    ],
    "L0-14": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14
    ]
  },
  "alphas": [
    0.5,
    1.0
  ],
  "selection": "fit split, most applying",
  "test": "dev + test splits (257)",
  "retention": "MBPP+ 378, cap 8,192",
  "control": "random unit direction per layer, seed 97, same layers and alpha"
}
```

## Results
```json
{
  "directions": {
    "pairs": 3583,
    "layers": {
      "0": {
        "escape_norm": 256.6606,
        "heldout_share_along": 0.3637,
        "heldout_mean_cos": 0.5349
      },
      "1": {
        "escape_norm": 71.3869,
        "heldout_share_along": 0.4009,
        "heldout_mean_cos": 0.5771
      },
      "2": {
        "escape_norm": 51.9369,
        "heldout_share_along": 0.3376,
        "heldout_mean_cos": 0.5499
      },
      "3": {
        "escape_norm": 25.55,
        "heldout_share_along": 0.2847,
        "heldout_mean_cos": 0.4684
      },
      "4": {
        "escape_norm": 7.0334,
        "heldout_share_along": 0.2647,
        "heldout_mean_cos": 0.3861
      },
      "5": {
        "escape_norm": 18.0539,
        "heldout_share_along": 0.2624,
        "heldout_mean_cos": 0.3552
      },
      "6": {
        "escape_norm": 40.7695,
        "heldout_share_along": 0.2854,
        "heldout_mean_cos": 0.3632
      },
      "7": {
        "escape_norm": 35.1075,
        "heldout_share_along": 0.205,
        "heldout_mean_cos": 0.3181
      },
      "8": {
        "escape_norm": 9.9099,
        "heldout_share_along": 0.1464,
        "heldout_mean_cos": 0.2859
      },
      "9": {
        "escape_norm": 2.8667,
        "heldout_share_along": 0.1198,
        "heldout_mean_cos": 0.2776
      },
      "10": {
        "escape_norm": 27.1774,
        "heldout_share_along": 0.1446,
        "heldout_mean_cos": 0.2763
      },
      "11": {
        "escape_norm": 22.1492,
        "heldout_share_along": 0.1283,
        "heldout_mean_cos": 0.2851
      },
      "12": {
        "escape_norm": 13.5723,
        "heldout_share_along": 0.1294,
        "heldout_mean_cos": 0.2796
      },
      "13": {
        "escape_norm": 6.3278,
        "heldout_share_along": 0.1432,
        "heldout_mean_cos": 0.3063
      },
      "14": {
        "escape_norm": 48.2788,
        "heldout_share_along": 0.1571,
        "heldout_mean_cos": 0.3471
      }
    }
  },
  "fit_scan": {
    "L13-14_a0.5": {
      "applies": 116,
      "correct": 23,
      "escape_errors": 16
    },
    "L13-14_a1.0": {
      "applies": 103,
      "correct": 29,
      "escape_errors": 7
    },
    "L0-14_a0.5": {
      "applies": 110,
      "correct": 31,
      "escape_errors": 11
    },
    "L0-14_a1.0": {
      "applies": 98,
      "correct": 24,
      "escape_errors": 8
    },
    "base": {
      "applies": 107,
      "correct": 24,
      "escape_errors": 22
    }
  },
  "selected": "L13-14_a0.5",
  "test": {
    "n": 257,
    "totals": {
      "base": {
        "applies": 99,
        "correct": 23,
        "esc": 32
      },
      "selected": {
        "applies": 88,
        "correct": 29,
        "esc": 22
      },
      "control": {
        "applies": 91,
        "correct": 21,
        "esc": 27
      }
    },
    "applies_mcnemar": {
      "selected_only": 27,
      "base_only": 38,
      "p": 0.2145
    }
  },
  "mbpp": {
    "n": 378,
    "base": 234,
    "selected": 230,
    "control": 227,
    "selected_only": 18,
    "base_only": 22,
    "p": 0.6358
  },
  "gate_passed": false,
  "headline": "escape-blind values L13-14_a0.5: test applies unedited 99/257, selected 88 (control 91), McNemar p = 0.2145; escape errors 32 -> 22; correct 23 -> 29; MBPP+ 234 -> 230 of 378; gate not passed",
  "system_monitor": {
    "samples": 147
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

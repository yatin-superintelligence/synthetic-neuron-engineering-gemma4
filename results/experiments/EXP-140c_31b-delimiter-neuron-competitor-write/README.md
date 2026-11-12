# EXP-140c: 31b-delimiter-neuron-competitor-write

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-11T06:37:03.379768Z / 2026-11-11T07:01:25.510408Z
- **Wall time:** 24.4 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-140, EXP-140b
- **Command:** `experiments/scripts/exp140c_31b_delimiter_competitors.py`

## Why this experiment
EXP-140b's minimum-norm write flipped 35 of 62 faulty states at its first strength and none at twice that, because a rare token rose faster than the closing token; the write must hold down the tokens that overtake it.

## Hypothesis (stated before running)
With the same read-out, the minimum-norm write makes the closing token the most probable at 70% or more of held-out faulty states while at most 1 in 1,000 other states change their most probable token.

## Configuration
```json
{
  "model": "google/gemma-4-31B-it-qat-q4_0-unquantized",
  "bench": "artifacts_local/EXP-140/delim_bench.jsonl.gz",
  "lambdas": [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0
  ],
  "write": "u = alpha g / |g|, g = e * w * w_f, e = E_close - mean(E_competitors)",
  "rounds": 6,
  "alpha_grid": "41 points, 0.1 to 20",
  "target_margin_p10": 2.0,
  "folds": {
    "a": [
      "fastapi_11194",
      "fastapi_14794",
      "fastapi_9753",
      "rich_3454",
      "rich_3470"
    ],
    "b": [
      "fastapi_14258",
      "rich_3067",
      "rich_3472",
      "rich_3944"
    ]
  },
  "gates": "held-out ERR flipped to the closing token >= 70%; held-out negatives with a changed top token <= 0.1%"
}
```

## Results
```json
{
  "fit": {
    "chosen": {
      "lam": 0.01,
      "pos_mean": 0.768,
      "hard_p99": 0.0983
    },
    "exactness": {
      "max_abs_logit_difference_recomputed_vs_model": 0.1921
    },
    "unedited": {
      "ERR": 62,
      "ERR_top_is_backtick_comma": 56,
      "ERR_top_is_close": 3
    },
    "direction": {
      "norm_ratio_old_to_min_for_equal_first_order_effect": 74.1
    },
    "old_write_recomputed": [
      {
        "c": 9.0,
        "close_is_top": 8,
        "margin_median": -3.02,
        "margin_p10": -11.85
      },
      {
        "c": 12.0,
        "close_is_top": 11,
        "margin_median": -2.56,
        "margin_p10": -10.66
      }
    ],
    "write": {
      "rounds": [
        {
          "round": 1,
          "competitors": [
            8347,
            236929
          ],
          "alpha": 1.8432,
          "close_is_top": 43,
          "n": 62,
          "margin_median": 0.32,
          "margin_p10": -2.47,
          "new_top_tokens": [
            [
              148,
              "            "
            ],
            [
              228276,
              "nilvadipine"
            ]
          ]
        },
        {
          "round": 2,
          "competitors": [
            8347,
            236929,
            148,
            228276
          ],
          "alpha": 4.0806,
          "close_is_top": 62,
          "n": 62,
          "margin_median": 2.42,
          "margin_p10": 1.56,
          "new_top_tokens": []
        }
      ],
      "competitors": [
        8347,
        236929,
        148,
        228276
      ],
      "alpha": 4.0806,
      "u_norm": 4.0806,
      "old_u_norm_at_c12": 66.2056,
      "in_sample": {
        "close_is_top": 62,
        "n": 62,
        "margin_median": 2.42,
        "margin_p10": 1.56
      }
    },
    "held_out_edit": {
      "by_class": {
        "CONT": {
          "n": 1686,
          "top_changed": 2,
          "top_is_close_before": 0,
          "top_is_close_after": 2,
          "control_top_changed": 0
        },
        "CONT_BT": {
          "n": 19,
          "top_changed": 0,
          "top_is_close_before": 0,
          "top_is_close_after": 0,
          "control_top_changed": 0
        },
        "CONT_H": {
          "n": 1416,
          "top_changed": 5,
          "top_is_close_before": 1,
          "top_is_close_after": 6,
          "control_top_changed": 1
        },
        "ERR": {
          "n": 62,
          "top_changed": 54,
          "top_is_close_before": 3,
          "top_is_close_after": 57,
          "control_top_changed": 2
        },
        "ERR2": {
          "n": 10,
          "top_changed": 0,
          "top_is_close_before": 0,
          "top_is_close_after": 0,
          "control_top_changed": 0
        },
        "OKC": {
          "n": 255,
          "top_changed": 6,
          "top_is_close_before": 248,
          "top_is_close_after": 254,
          "control_top_changed": 0
        },
        "OPEN": {
          "n": 255,
          "top_changed": 0,
          "top_is_close_before": 255,
          "top_is_close_after": 255,
          "control_top_changed": 0
        },
        "OTHER": {
          "n": 870,
          "top_changed": 1,
          "top_is_close_before": 0,
          "top_is_close_after": 0,
          "control_top_changed": 6
        },
        "TXT_BT": {
          "n": 3944,
          "top_changed": 3,
          "top_is_close_before": 0,
          "top_is_close_after": 0,
          "control_top_changed": 11
        },
        "TXT_BTC": {
          "n": 141,
          "top_changed": 0,
          "top_is_close_before": 0,
          "top_is_close_after": 0,
          "control_top_changed": 0
        }
      },
      "ERR_flipped_to_close": 57,
      "ERR_n": 62,
      "negatives_changed": 11,
      "negatives_n": 8076,
      "control_ERR_flipped": 3,
      "control_negatives_changed": 18,
      "readout_held": {
        "ERR_mean": 0.768,
        "neg_p99": 0.1028
      }
    },
    "gates": {
      "ERR_flipped_at_least_70_percent": true,
      "negatives_changed_at_most_1_in_1000": false
    }
  },
  "generation": {
    "states": 16,
    "unedited_repeats_the_served_token": 12,
    "unedited_well_formed": 2,
    "edited_first_token_is_close": 14,
    "edited_well_formed": 13
  },
  "adapters": [],
  "headline": "31B delimiter neuron with the competitor write (gates not passed, not saved): held-out, the closing token is the most probable at 57 of 62 faulty states (control 3); 11 of 8076 other states change (control 18); greedy continuations well formed 13 of 16 edited against 2 unedited; write norm 4.0806 against 66.2056",
  "system_monitor": {
    "samples": 25
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

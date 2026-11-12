# EXP-087: weight-extrapolation

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T15:22:16.093620Z / 2026-11-08T16:55:43.986640Z
- **Wall time:** 93.5 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-086
- **Command:** `experiments/scripts/exp087_weight_extrapolation.py`

## Why this experiment
Training-free ways to add capability found by the 2026-11-08 literature survey; ExPO needs only the public base and instruct checkpoints, and its result is one weight delta.

## Hypothesis (stated before running)
Some alpha in 0.1 to 0.5 raises E2B's MBPP+ test-half pass count against alpha = 0 (exact McNemar p < 0.05) and against a control edit of matched strength, and holds at FP8.

## Configuration
```json
{
  "alphas": [
    0.1,
    0.2,
    0.3,
    0.5
  ],
  "repos": {
    "qat": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    "it": "google/gemma-4-E2B-it",
    "base": "google/gemma-4-E2B"
  },
  "scan": "MBPP+ dev 189, cap 4,096",
  "confirm": "MBPP+ test 189, 8,192, bf16 and FP8",
  "control": "random direction per tensor, same Frobenius norm as alpha * delta, seed 87",
  "gate": "McNemar p < 0.05 on test at bf16, above control, not lower at FP8"
}
```

## Results
```json
{
  "scan": {
    "alpha0": [
      121,
      {
        "pass": 121,
        "wrong_answer": 43,
        "wrong_or_missing_entry_point": 12,
        "runtime_error": 11,
        "other_fail": 1,
        "timeout": 1
      }
    ],
    "alpha0.1": [
      85,
      {
        "pass": 85,
        "wrong_or_missing_entry_point": 42,
        "syntax_error": 6,
        "wrong_answer": 37,
        "name_or_import_error": 4,
        "runtime_error": 13,
        "other_fail": 1,
        "truncated": 1
      }
    ],
    "alpha0.2": [
      16,
      {
        "wrong_or_missing_entry_point": 75,
        "syntax_error": 21,
        "loop": 31,
        "wrong_answer": 20,
        "truncated": 7,
        "pass": 16,
        "name_or_import_error": 9,
        "runtime_error": 7,
        "timeout": 3
      }
    ],
    "alpha0.3": [
      0,
      {
        "wrong_or_missing_entry_point": 72,
        "loop": 90,
        "truncated": 9,
        "syntax_error": 12,
        "name_or_import_error": 3,
        "wrong_answer": 2,
        "runtime_error": 1
      }
    ],
    "alpha0.5": [
      0,
      {
        "no_code_block": 152,
        "wrong_or_missing_entry_point": 34,
        "syntax_error": 2,
        "loop": 1
      }
    ]
  },
  "builds": {
    "alpha0": {
      "moved": 540,
      "kept": 1411,
      "missing": [],
      "k_norm_placeholders_added": 20,
      "rel_change_median": 0.0,
      "rel_change_max": 0.0
    },
    "alpha0.1": {
      "moved": 540,
      "kept": 1411,
      "missing": [],
      "k_norm_placeholders_added": 20,
      "rel_change_median": 0.03495,
      "rel_change_max": 3.56746
    },
    "alpha0.2": {
      "moved": 540,
      "kept": 1411,
      "missing": [],
      "k_norm_placeholders_added": 20,
      "rel_change_median": 0.06989,
      "rel_change_max": 7.13492
    },
    "alpha0.3": {
      "moved": 540,
      "kept": 1411,
      "missing": [],
      "k_norm_placeholders_added": 20,
      "rel_change_median": 0.10484,
      "rel_change_max": 10.70238
    },
    "alpha0.5": {
      "moved": 540,
      "kept": 1411,
      "missing": [],
      "k_norm_placeholders_added": 20,
      "rel_change_median": 0.17473,
      "rel_change_max": 17.8373
    },
    "control": {
      "moved": 540,
      "kept": 1411,
      "missing": [],
      "k_norm_placeholders_added": 20,
      "rel_change_median": 0.03495,
      "rel_change_max": 3.56746
    }
  },
  "selected": "alpha0.1",
  "test_bf16": {
    "alpha0": 112,
    "alpha0.1": 101,
    "control": 0
  },
  "test_fp8": {
    "alpha0_fp8": 116,
    "alpha0.1_fp8": 88
  },
  "stats": {
    "best_only": 21,
    "alpha0_only": 32,
    "p": 0.169,
    "best_vs_control": {
      "best_only": 101,
      "control_only": 0,
      "p": 0.0
    },
    "fp8": {
      "best_only": 12,
      "alpha0_only": 40,
      "p": 0.0001
    }
  },
  "gate_passed": false,
  "headline": "selected alpha0.1: dev 85 vs 121 of 189; test bf16 101 vs 112 (control 0) of 189, McNemar p = 0.169; gate not passed",
  "system_monitor": {
    "samples": 104
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

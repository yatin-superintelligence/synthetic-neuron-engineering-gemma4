# EXP-092: patch-correctness-neurons

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T19:15:31.168352Z / 2026-11-08T21:42:24.423860Z
- **Wall time:** 146.9 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-089, EXP-088
- **Command:** `experiments/scripts/exp092_patch_neurons.py`

## Why this experiment
The agent mostly fails by writing wrong patches; a verified bench of fix states measures that step directly, and a neuron built from E2B's own correct-minus-incorrect edits targets it without training.

## Hypothesis (stated before running)
A patch-correctness neuron raises E2B's correct-edit rate on held-out bench states against the unedited model (exact McNemar p < 0.05) and against a control neuron of matched strength.

## Configuration
```json
{
  "layers": [
    9,
    12,
    15,
    18,
    21,
    24,
    27
  ],
  "alphas": [
    1,
    3,
    9
  ],
  "splits": "fit 50 / dev 25 / test 25, seed 92",
  "samples_per_fit_state": 4,
  "fit_temperature": 1.0,
  "answer_tokens": "2,048 then thought closed and 1,024 more",
  "model": "official E2B W4A16 on vLLM 0.19.1"
}
```

## Results
```json
{
  "states": 514,
  "skipped": {
    "no single-line fix": 49
  },
  "baseline": {
    "fit": {
      "correct": 28,
      "applies": 102,
      "n": 257
    },
    "dev": {
      "correct": 8,
      "applies": 45,
      "n": 128
    },
    "test": {
      "correct": 11,
      "applies": 39,
      "n": 129
    }
  },
  "next_tool": {
    "edit_file": 284,
    "null": 47,
    "read_file": 106,
    "run_command": 30,
    "search_similar_code": 40,
    "submit_patch": 7
  },
  "directions": {
    "samples": 968,
    "mixed_states": 41,
    "held_mixed": 13,
    "layers": {
      "9": {
        "pairwise_acc": 0.6923,
        "dmu": 3.7559
      },
      "12": {
        "pairwise_acc": 0.6923,
        "dmu": 3.2909
      },
      "15": {
        "pairwise_acc": 0.8462,
        "dmu": 6.2969,
        "readout_check": {
          "pos_mean": 0.928,
          "pos_min": 0.049,
          "neg_median": 0.0124,
          "neg_p99": 0.3782,
          "neg_p999": 0.5909,
          "neg_max": 0.7071,
          "n_pos": 1066,
          "n_neg": 1200
        },
        "rms_y": 0.1698,
        "first_order_shift": 5.6569,
        "target": 6.2969,
        "control_first_order": 5.6073
      },
      "18": {
        "pairwise_acc": 0.7692,
        "dmu": 5.3651
      },
      "21": {
        "pairwise_acc": 0.7692,
        "dmu": 4.1574
      },
      "24": {
        "pairwise_acc": 0.7692,
        "dmu": 3.9793
      },
      "27": {
        "pairwise_acc": 0.8462,
        "dmu": 3.1334,
        "readout_check": {
          "pos_mean": 0.843,
          "pos_min": 0.133,
          "neg_median": 0.0806,
          "neg_p99": 0.7738,
          "neg_p999": 0.8924,
          "neg_max": 0.9115,
          "n_pos": 1066,
          "n_neg": 1200
        },
        "rms_y": 0.2357,
        "first_order_shift": 1.3249,
        "target": 3.1334,
        "control_first_order": 1.0867
      }
    },
    "selected_layers": [
      15,
      27
    ]
  },
  "dev_scan": {
    "n_L15_a1": 11,
    "n_L15_a3": 6,
    "n_L15_a9": 8,
    "n_L27_a1": 6,
    "n_L27_a3": 6,
    "n_L27_a9": 7,
    "base": 8
  },
  "selected": "n_L15_a1",
  "test": {
    "n": 129,
    "base": 11,
    "selected": 12,
    "control": 14,
    "selected_only": 6,
    "base_only": 5,
    "p": 1.0,
    "vs_control": {
      "selected_only": 3,
      "control_only": 5,
      "p": 0.7266
    },
    "applies": {
      "selected": 50,
      "control": 49
    }
  },
  "gate_passed": false,
  "headline": "patch bench: unedited correct 11/129 on test; n_L15_a1 12 (control 14), McNemar p = 1.0; gate not passed",
  "system_monitor": {
    "samples": 249
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

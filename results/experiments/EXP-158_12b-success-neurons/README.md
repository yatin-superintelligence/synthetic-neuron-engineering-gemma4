# EXP-158: 12b-success-neurons

- **Status:** OK
- **Category:** diagnosis
- **Started / finished (UTC):** 2026-11-12T11:45:31.974498Z / 2026-11-12T11:49:50.481624Z
- **Wall time:** 4.3 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-147
- **Command:** `/content/repo/experiments/scripts/exp154_31b_success_neurons.py`

## Why this experiment
The author asked which neurons contribute to solving tasks. EXP-147 read the residual stream at 8 depths; this scans every MLP neuron of 31B at the fix, with selection and test on different tasks.

## Hypothesis (stated before running)
64 neurons chosen on one half of the tasks rank a passing run above a failing run of the same task on the other half with AUC 0.70 or more, permutation p < 0.05 with the selection repeated under shuffled labels.

## Configuration
```json
{
  "model": "google/gemma-4-12B-it-qat-q4_0-unquantized",
  "bench": "artifacts_local/EXP-158/success_bench.jsonl.gz",
  "k": 64,
  "permutations": 200
}
```

## Results
```json
{
  "headline": "64 neurons chosen on half the tasks: held-out AUC 0.7083 (null median 0.5, p 0.1343); gate not passed",
  "fit": {
    "rows": 62,
    "solved": 44,
    "tasks_with_both": 11,
    "features": 1474560,
    "k": 64,
    "held_out": {
      "a_to_b": 0.7,
      "b_to_a": 0.7143,
      "pooled": 0.7083,
      "pairs": 24
    },
    "permutation": {
      "n": 200,
      "null_median": 0.5,
      "null_95": 0.75,
      "p": 0.1343
    },
    "gate_passed": false,
    "top10": [
      {
        "feature": 83166,
        "kind": "last",
        "layer": 5,
        "neuron": 6366,
        "auc": 0.0,
        "mean_solved": -0.03122,
        "mean_failed": -0.02425
      },
      {
        "feature": 156886,
        "kind": "last",
        "layer": 10,
        "neuron": 3286,
        "auc": 0.0,
        "mean_solved": 0.0241,
        "mean_failed": 0.05758
      },
      {
        "feature": 273679,
        "kind": "last",
        "layer": 17,
        "neuron": 12559,
        "auc": 0.0,
        "mean_solved": 0.12124,
        "mean_failed": 0.14798
      },
      {
        "feature": 340437,
        "kind": "last",
        "layer": 22,
        "neuron": 2517,
        "auc": 0.0,
        "mean_solved": 0.22997,
        "mean_failed": 0.37598
      },
      {
        "feature": 352421,
        "kind": "last",
        "layer": 22,
        "neuron": 14501,
        "auc": 0.0,
        "mean_solved": 0.01029,
        "mean_failed": 0.02774
      },
      {
        "feature": 379465,
        "kind": "last",
        "layer": 24,
        "neuron": 10825,
        "auc": 1.0,
        "mean_solved": -0.04666,
        "mean_failed": -0.0805
      },
      {
        "feature": 381514,
        "kind": "last",
        "layer": 24,
        "neuron": 12874,
        "auc": 0.0,
        "mean_solved": 0.04035,
        "mean_failed": 0.06676
      },
      {
        "feature": 382043,
        "kind": "last",
        "layer": 24,
        "neuron": 13403,
        "auc": 1.0,
        "mean_solved": 0.00613,
        "mean_failed": -0.00389
      },
      {
        "feature": 430510,
        "kind": "last",
        "layer": 28,
        "neuron": 430,
        "auc": 0.0,
        "mean_solved": -0.01189,
        "mean_failed": 0.00051
      },
      {
        "feature": 443615,
        "kind": "last",
        "layer": 28,
        "neuron": 13535,
        "auc": 1.0,
        "mean_solved": -0.01938,
        "mean_failed": -0.03884
      }
    ],
    "top_layers": {
      "3": 1,
      "4": 2,
      "5": 4,
      "6": 2,
      "7": 3,
      "8": 1,
      "9": 1,
      "10": 1,
      "12": 1,
      "15": 1,
      "16": 2,
      "17": 2,
      "21": 2,
      "22": 2,
      "24": 3,
      "27": 2,
      "28": 3,
      "31": 1,
      "33": 2,
      "34": 2,
      "35": 3,
      "36": 4,
      "37": 4,
      "38": 2,
      "39": 2,
      "40": 2,
      "41": 1,
      "42": 1,
      "43": 3,
      "44": 1,
      "45": 2,
      "46": 1
    }
  },
  "system_monitor": {
    "samples": 5
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

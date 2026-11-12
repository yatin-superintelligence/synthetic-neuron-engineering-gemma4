# EXP-154: 31b-success-neurons

- **Status:** OK
- **Category:** diagnosis
- **Started / finished (UTC):** 2026-11-12T05:53:34.654908Z / 2026-11-12T06:10:09.039516Z
- **Wall time:** 16.6 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-147
- **Command:** `experiments/scripts/exp154_31b_success_neurons.py`

## Why this experiment
The author asked which neurons contribute to solving tasks. EXP-147 read the residual stream at 8 depths; this scans every MLP neuron of 31B at the fix, with selection and test on different tasks.

## Hypothesis (stated before running)
64 neurons chosen on one half of the tasks rank a passing run above a failing run of the same task on the other half with AUC 0.70 or more, permutation p < 0.05 with the selection repeated under shuffled labels.

## Configuration
```json
{
  "model": "google/gemma-4-31B-it-qat-q4_0-unquantized",
  "bench": "artifacts_local/EXP-147/success_bench.jsonl.gz",
  "k": 64,
  "permutations": 200
}
```

## Results
```json
{
  "headline": "64 neurons chosen on half the tasks: held-out AUC 0.6506 (null median 0.4834, p 0.0348); gate not passed",
  "fit": {
    "rows": 123,
    "solved": 85,
    "tasks_with_both": 13,
    "features": 2580480,
    "k": 64,
    "held_out": {
      "a_to_b": 0.7027,
      "b_to_a": 0.6087,
      "pooled": 0.6506,
      "pairs": 166
    },
    "permutation": {
      "n": 200,
      "null_median": 0.4834,
      "null_95": 0.6386,
      "p": 0.0348
    },
    "gate_passed": false,
    "top10": [
      {
        "feature": 1727894,
        "kind": "outmean",
        "layer": 20,
        "neuron": 7574,
        "auc": 0.1446,
        "mean_solved": -0.0173,
        "mean_failed": -0.00587
      },
      {
        "feature": 716209,
        "kind": "last",
        "layer": 33,
        "neuron": 6577,
        "auc": 0.8554,
        "mean_solved": -0.0521,
        "mean_failed": -0.10412
      },
      {
        "feature": 1333546,
        "kind": "outmean",
        "layer": 2,
        "neuron": 298,
        "auc": 0.8494,
        "mean_solved": 0.01675,
        "mean_failed": 0.01266
      },
      {
        "feature": 1388618,
        "kind": "outmean",
        "layer": 4,
        "neuron": 12362,
        "auc": 0.8494,
        "mean_solved": 0.07067,
        "mean_failed": 0.05021
      },
      {
        "feature": 1093053,
        "kind": "last",
        "layer": 50,
        "neuron": 17853,
        "auc": 0.1566,
        "mean_solved": -0.01235,
        "mean_failed": 0.02993
      },
      {
        "feature": 862411,
        "kind": "last",
        "layer": 40,
        "neuron": 2251,
        "auc": 0.1627,
        "mean_solved": 0.00165,
        "mean_failed": 0.02016
      },
      {
        "feature": 1701479,
        "kind": "outmean",
        "layer": 19,
        "neuron": 2663,
        "auc": 0.1627,
        "mean_solved": 0.02736,
        "mean_failed": 0.03319
      },
      {
        "feature": 2098378,
        "kind": "outmean",
        "layer": 37,
        "neuron": 12490,
        "auc": 0.1627,
        "mean_solved": -0.00565,
        "mean_failed": -0.00131
      },
      {
        "feature": 721063,
        "kind": "last",
        "layer": 33,
        "neuron": 11431,
        "auc": 0.8373,
        "mean_solved": 0.11097,
        "mean_failed": 0.01816
      },
      {
        "feature": 1057196,
        "kind": "last",
        "layer": 49,
        "neuron": 3500,
        "auc": 0.8373,
        "mean_solved": -0.00478,
        "mean_failed": -0.01245
      }
    ],
    "top_layers": {
      "0": 1,
      "2": 2,
      "4": 2,
      "6": 1,
      "7": 1,
      "8": 1,
      "9": 4,
      "10": 1,
      "14": 2,
      "17": 1,
      "18": 1,
      "19": 3,
      "20": 4,
      "21": 1,
      "23": 1,
      "24": 1,
      "27": 1,
      "30": 1,
      "32": 3,
      "33": 3,
      "34": 2,
      "35": 3,
      "37": 1,
      "38": 2,
      "39": 1,
      "40": 1,
      "41": 2,
      "43": 1,
      "45": 2,
      "47": 3,
      "49": 3,
      "50": 1,
      "51": 1,
      "53": 1,
      "54": 1,
      "56": 1,
      "57": 1,
      "58": 2
    }
  },
  "system_monitor": {
    "samples": 17
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

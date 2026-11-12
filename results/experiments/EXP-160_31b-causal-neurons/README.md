# EXP-160: 31b-causal-neurons

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-12T12:14:14.182245Z / 2026-11-12T12:23:53.504914Z
- **Wall time:** 9.7 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-147, EXP-154
- **Command:** `experiments/scripts/exp160_31b_causal_neurons.py`

## Why this experiment
Neurons chosen by correlation with success mark it but are not levers; the author asked for the neurons responsible for better performance, which needs an interventional search.

## Hypothesis (stated before running)
Neurons chosen on half the tasks by their attributed effect on 31B's preference for a passing run's edit over its own failing edit, when actually scaled, raise that preference in at least 65% of held-out pairs and more than random neurons do.

## Configuration
```json
{
  "model": "google/gemma-4-31B-it-qat-q4_0-unquantized",
  "bench": "artifacts_local/EXP-147/success_bench.jsonl.gz",
  "k": 32,
  "alpha": 1.0,
  "max_call_tokens": 64,
  "pairs_per_fail": 1
}
```

## Results
```json
{
  "headline": "held out: chosen neurons raise the preference in 7 of 7 pairs (random 4); mean change 0.1233 against 0.0051; gate passed",
  "fit": {
    "pairs": 7,
    "tasks": 5,
    "fold_a": [
      "fastapi_9753",
      "fastapi_13537",
      "rich_3480"
    ],
    "fold_b": [
      "rich_3470",
      "fastapi_15589"
    ],
    "preference_mean": -0.4564,
    "held_out": {
      "pairs": 7,
      "chosen_raises": 7,
      "chosen_mean_change": 0.1233,
      "random_raises": 4,
      "random_mean_change": 0.0051,
      "chosen_minus_random_mean": 0.1183,
      "chosen_beats_random": 7
    },
    "gate_passed": true,
    "chosen_layers": [
      25,
      28,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      40,
      41,
      42,
      43,
      44,
      45,
      48,
      49,
      50,
      51,
      52,
      54,
      55,
      56,
      57,
      58,
      59
    ],
    "top5": [
      {
        "layer": 36,
        "neuron": 1446,
        "sign": 1,
        "attribution": 0.021768
      },
      {
        "layer": 51,
        "neuron": 17235,
        "sign": 1,
        "attribution": 0.019089
      },
      {
        "layer": 38,
        "neuron": 10587,
        "sign": 1,
        "attribution": 0.018472
      },
      {
        "layer": 41,
        "neuron": 20150,
        "sign": 1,
        "attribution": 0.016854
      },
      {
        "layer": 54,
        "neuron": 16223,
        "sign": 1,
        "attribution": 0.016551
      }
    ]
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

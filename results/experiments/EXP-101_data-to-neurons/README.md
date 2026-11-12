# EXP-101: data-to-neurons

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-09T09:29:42.192810Z / 2026-11-09T09:37:53.796613Z
- **Wall time:** 8.2 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-092, EXP-100
- **Command:** `experiments/scripts/exp101_data_to_neurons.py`

## Why this experiment
The author's data-to-neuron idea: the fit tasks' correct fixes stored in one MLP layer by a closed-form least-squares memory.

## Hypothesis (stated before running)
A closed-form memory of the fit tasks' correct edits raises correct answers on held-out bench states against the unedited model and against the same memory built from shuffled values.

## Configuration
```json
{
  "layers": [
    12,
    15
  ],
  "c": [
    2.0,
    4.0
  ],
  "rank": 64,
  "control": "values shuffled across positions"
}
```

## Results
```json
{
  "memory": {
    "fit_states": 257,
    "target_positions": 6983,
    "positions_where_e2b_is_wrong": 692,
    "layers": {
      "mem_L12_c2": {
        "rank_energy_share": 0.9997,
        "delta_norm": 8.174,
        "relative_to_W": 0.21466
      },
      "mem_L12_c4": {
        "rank_energy_share": 0.9997,
        "delta_norm": 16.348,
        "relative_to_W": 0.42933
      },
      "shuf_L12_c4": {
        "rank_energy_share": 0.9997,
        "delta_norm": 17.252,
        "relative_to_W": 0.45309
      },
      "mem_L15_c2": {
        "rank_energy_share": 1.0,
        "delta_norm": 1.613,
        "relative_to_W": 0.04169
      },
      "mem_L15_c4": {
        "rank_energy_share": 1.0,
        "delta_norm": 3.226,
        "relative_to_W": 0.08339
      },
      "shuf_L15_c4": {
        "rank_energy_share": 1.0,
        "delta_norm": 3.299,
        "relative_to_W": 0.08529
      }
    }
  },
  "adapters": [
    "mem_L12_c2",
    "mem_L12_c4",
    "mem_L15_c2",
    "mem_L15_c4",
    "shuf_L12_c4",
    "shuf_L15_c4"
  ],
  "headline": "data-to-neuron memory from 257 fit tasks: 6983 target positions, E2B wrong at 692; adapters ['mem_L12_c2', 'mem_L12_c4', 'mem_L15_c2', 'mem_L15_c4', 'shuf_L12_c4', 'shuf_L15_c4']",
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

# EXP-152: 31b-submit-readout

- **Status:** OK
- **Category:** diagnosis
- **Started / finished (UTC):** 2026-11-11T21:27:20.244014Z / 2026-11-11T21:41:37.479933Z
- **Wall time:** 14.3 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-147
- **Command:** `experiments/scripts/exp152_31b_submit_readout.py`

## Why this experiment
EXP-147 found no success signal at the first real edit; at the submit call the model has seen its own tests, so a signal there would allow a gated keep-working unit.

## Hypothesis (stated before running)
Held out by task, a task-centred mean-difference read-out at some depth ranks a run that will pass above one that will not (same task) with pooled AUC 0.70 or more, permutation p < 0.05 after selection over depths and positions.

## Configuration
```json
{
  "model": "google/gemma-4-31B-it-qat-q4_0-unquantized",
  "bench": "artifacts_local/EXP-152/submit_bench.jsonl.gz",
  "fractions": [
    0.25,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
  ],
  "samples_per_row": 8,
  "permutations": 300
}
```

## Results
```json
{
  "headline": "held-out within-task AUC 0.528 at layer 29 (outmean), permutation p 0.9336; gate not passed",
  "fit": {
    "rows": 122,
    "solved": 85,
    "tasks": {
      "fastapi_13537": "7 of 7",
      "fastapi_14463": "6 of 7",
      "fastapi_15589": "5 of 8",
      "fastapi_5624": "3 of 7",
      "fastapi_9425": "8 of 8",
      "fastapi_9753": "7 of 11",
      "rich_2725": "7 of 8",
      "rich_3043": "6 of 6",
      "rich_3063": "3 of 5",
      "rich_3067": "11 of 12",
      "rich_3278": "6 of 7",
      "rich_3470": "3 of 13",
      "rich_3480": "6 of 8",
      "rich_4077": "4 of 8",
      "rich_4079": "3 of 7"
    },
    "layers": [
      14,
      23,
      29,
      35,
      41,
      47,
      53,
      59
    ],
    "auc": {
      "L14 last": 0.385,
      "L14 outmean": 0.466,
      "L23 last": 0.404,
      "L23 outmean": 0.484,
      "L29 last": 0.429,
      "L29 outmean": 0.528,
      "L35 last": 0.447,
      "L35 outmean": 0.46,
      "L41 last": 0.522,
      "L41 outmean": 0.422,
      "L47 last": 0.491,
      "L47 outmean": 0.453,
      "L53 last": 0.497,
      "L53 outmean": 0.466,
      "L59 last": 0.509,
      "L59 outmean": 0.416
    },
    "best": {
      "layer": 29,
      "position": "outmean",
      "auc": 0.528
    },
    "permutation": {
      "n": 300,
      "null_max_auc_median": 0.615,
      "null_max_auc_95": 0.69,
      "p": 0.9336
    },
    "gate_passed": false
  },
  "system_monitor": {
    "samples": 15
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

# EXP-147: 31b-success-readout

- **Status:** OK
- **Category:** diagnosis
- **Started / finished (UTC):** 2026-11-11T20:42:22.092960Z / 2026-11-11T20:59:13.550568Z
- **Wall time:** 16.9 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-139B, EXP-144a, EXP-145a
- **Command:** `experiments/scripts/exp147_31b_success_readout.py`

## Why this experiment
31B's remaining failures on solvable tasks are wrong fixes on tasks it solves only sometimes; an edit can add solves there only if the model's state at the fix already separates fixes that will pass from fixes that will not.

## Hypothesis (stated before running)
Held out by task, a task-centred mean-difference read-out at some depth ranks a run that will pass above one that will not (same task) with pooled AUC 0.70 or more, permutation p < 0.05 after selection over depths and positions.

## Configuration
```json
{
  "model": "google/gemma-4-31B-it-qat-q4_0-unquantized",
  "bench": "artifacts_local/EXP-147/success_bench.jsonl.gz",
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
  "headline": "held-out within-task AUC 0.633 at layer 23 (last), permutation p 0.3787; gate not passed",
  "fit": {
    "rows": 123,
    "solved": 85,
    "tasks": {
      "fastapi_13537": "7 of 8",
      "fastapi_14463": "7 of 8",
      "fastapi_15589": "5 of 8",
      "fastapi_5624": "3 of 7",
      "fastapi_9425": "5 of 5",
      "fastapi_9753": "7 of 8",
      "rich_2725": "7 of 8",
      "rich_3043": "7 of 7",
      "rich_3063": "3 of 7",
      "rich_3067": "11 of 13",
      "rich_3278": "7 of 8",
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
      "L14 last": 0.536,
      "L14 outmean": 0.566,
      "L23 last": 0.633,
      "L23 outmean": 0.627,
      "L29 last": 0.542,
      "L29 outmean": 0.59,
      "L35 last": 0.542,
      "L35 outmean": 0.578,
      "L41 last": 0.572,
      "L41 outmean": 0.602,
      "L47 last": 0.584,
      "L47 outmean": 0.59,
      "L53 last": 0.59,
      "L53 outmean": 0.572,
      "L59 last": 0.596,
      "L59 outmean": 0.614
    },
    "best": {
      "layer": 23,
      "position": "last",
      "auc": 0.633
    },
    "permutation": {
      "n": 300,
      "null_max_auc_median": 0.608,
      "null_max_auc_95": 0.693,
      "p": 0.3787
    },
    "gate_passed": false
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

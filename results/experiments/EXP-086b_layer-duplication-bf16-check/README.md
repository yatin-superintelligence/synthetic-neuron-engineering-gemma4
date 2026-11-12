# EXP-086b: layer-duplication-bf16-check

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T17:18:38.451618Z / 2026-11-08T17:41:00.002219Z
- **Wall time:** 22.4 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-086
- **Command:** `experiments/scripts/exp086b_bf16_check.py`

## Why this experiment
EXP-086's pre-registered bf16 check could not load (missing k_norm in KV-shared layers of the public bf16 checkpoint).

## Hypothesis (stated before running)
As pre-registered in D-056: the selected block is not lower than the unedited model on MBPP+ test at bf16.

## Configuration
```json
{
  "fix": "k_norm of KV-shared layers copied from the official W4A16 checkpoint"
}
```

## Results
```json
{
  "selected": "dup_s31_k2",
  "test_bf16": {
    "official": 113,
    "dup_s31_k2": 103
  },
  "stats": {
    "best_only": 13,
    "official_only": 23,
    "p": 0.1325
  },
  "headline": "bf16 MBPP+ test: dup_s31_k2 103 vs unedited 113 of 189 (McNemar p = 0.1325)",
  "system_monitor": {
    "samples": 23
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

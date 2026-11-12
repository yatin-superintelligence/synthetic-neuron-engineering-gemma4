# EXP-110b: success-distillation-train-test

- **Status:** OK
- **Category:** training
- **Started / finished (UTC):** 2026-11-10T18:47:53.472477Z / 2026-11-10T19:33:06.953086Z
- **Wall time:** 45.2 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-088, EXP-103
- **Command:** `/content/repo/experiments/scripts/exp110_success_distillation.py`

## Why this experiment
Fine-tune E2B on real, officially resolved agent runs (its own and larger models') from one half of the 58 tasks; evaluate on the other half only.

## Hypothesis (stated before running)
Trained on the other half's successful runs, E2B solves more tasks of this half than unedited E2B at the same seed.

## Configuration
```json
{
  "half": "test",
  "cap_per_task": 8,
  "tool_chars": 1500,
  "rank": 16,
  "alpha": 32,
  "lr": 0.0001,
  "epochs": 2,
  "max_len": 24576
}
```

## Results
```json
{
  "half": "test",
  "tasks": 14,
  "trajectories": 40,
  "train": 35,
  "val": 1,
  "history": {
    "val_before": 0.4796,
    "train": [],
    "val": []
  },
  "vllm_check": {
    "base": "(a, b), (a,",
    "sft": " (a, b), (a,"
  },
  "adapter": "/content/repo/artifacts_local/EXP-110b/sft_lora",
  "headline": "SFT on 35 resolved runs of the test half: validation loss 0.4796 -> None",
  "system_monitor": {
    "samples": 46
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

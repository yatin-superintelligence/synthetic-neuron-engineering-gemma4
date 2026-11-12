# EXP-088b: sft-edit-only

- **Status:** OK
- **Category:** training
- **Started / finished (UTC):** 2026-11-08T22:36:05.807576Z / 2026-11-09T01:14:03.004827Z
- **Wall time:** 158.0 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-088, EXP-088a
- **Command:** `experiments/scripts/exp088b_sft_edit_only.py`

## Why this experiment
EXP-088's whole-trajectory SFT taught a bad exploring policy (0 of 56 solved) but unescaped old_strings; training only the edit turn keeps the fix-writing skill without changing how the agent explores.

## Hypothesis (stated before running)
An SFT LoRA trained only on the edit_file turn raises E2B's official tasks solved at seed 1 against unedited E2B at seed 1 on the same hardware, with fewer failed edits.

## Configuration
```json
{
  "rank": 16,
  "alpha": 32,
  "lr": 0.0001,
  "epochs": 2,
  "accumulation": 8,
  "warmup": 20,
  "max_len": 12288,
  "target": ".*language_model\\.layers\\.\\d+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.o_proj)$",
  "base": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
  "loss": "edit_file turn only",
  "context": "trajectory up to and including the edit turn"
}
```

## Results
```json
{
  "tasks": 563,
  "trajectories": 514,
  "skipped": {
    "no single-line edit": 49
  },
  "train": 466,
  "val": 24,
  "history": {
    "val_before": 0.967,
    "train": [
      [
        10,
        0.9126
      ],
      [
        20,
        0.3288
      ],
      [
        30,
        0.123
      ],
      [
        40,
        0.0706
      ],
      [
        50,
        0.0593
      ],
      [
        60,
        0.052
      ],
      [
        70,
        0.0396
      ],
      [
        80,
        0.0319
      ],
      [
        90,
        0.0273
      ],
      [
        100,
        0.0333
      ],
      [
        110,
        0.0347
      ]
    ],
    "val": [
      [
        50,
        0.0427
      ],
      [
        100,
        0.0292
      ]
    ]
  },
  "vllm_check": {
    "base": "(a, b), (a,",
    "sft": "\n```\n```\n```\n```"
  },
  "adapter": "/content/repo/artifacts_local/EXP-088b/sft_lora",
  "headline": "SFT (edit turn only) LoRA r16 on 466 examples: validation loss 0.967 -> 0.0292",
  "system_monitor": {
    "samples": 158
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

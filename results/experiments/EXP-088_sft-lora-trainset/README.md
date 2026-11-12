# EXP-088: sft-lora-trainset

- **Status:** OK
- **Category:** training
- **Started / finished (UTC):** 2026-11-08T16:39:44.988411Z / 2026-11-08T19:28:23.519680Z
- **Wall time:** 168.6 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-044a
- **Command:** `experiments/scripts/exp088_sft_lora.py`

## Why this experiment
Training track requested by the author: an SFT LoRA on verified bug-fix tasks, as a trained comparison for the training-free edits.

## Hypothesis (stated before running)
SFT on scripted trajectories over at least 500 verified injected-bug tasks raises E2B's official tasks solved at seeds 1 to 3 against unedited E2B at the same seeds (task-level sign-flip p < 0.05).

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
  "base": "google/gemma-4-E2B-it-qat-q4_0-unquantized"
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
  "train": 464,
  "val": 24,
  "history": {
    "val_before": 1.1735,
    "train": [
      [
        10,
        1.0962
      ],
      [
        20,
        0.5012
      ],
      [
        30,
        0.1929
      ],
      [
        40,
        0.1127
      ],
      [
        50,
        0.0954
      ],
      [
        60,
        0.0903
      ],
      [
        70,
        0.0749
      ],
      [
        80,
        0.0765
      ],
      [
        90,
        0.072
      ],
      [
        100,
        0.0686
      ],
      [
        110,
        0.0622
      ]
    ],
    "val": [
      [
        50,
        0.0878
      ],
      [
        100,
        0.0683
      ],
      [
        116,
        0.0682
      ]
    ]
  },
  "vllm_check": {
    "base": "(a, b), (a,",
    "sft": "(a, b), (a,"
  },
  "adapter": "/content/repo/artifacts_local/EXP-088/sft_lora",
  "headline": "SFT LoRA r16 on 464 trajectories: validation loss 1.1735 -> 0.0682",
  "system_monitor": {
    "samples": 169
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

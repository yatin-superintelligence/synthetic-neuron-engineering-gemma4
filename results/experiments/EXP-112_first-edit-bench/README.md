# EXP-112: first-edit-bench

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-09T20:22:34.617573Z / 2026-11-09T23:35:06.947904Z
- **Wall time:** 192.5 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-104, EXP-099, EXP-080, EXP-102, EXP-111
- **Command:** `experiments/scripts/exp112_first_edit_bench.py`

## Why this experiment
The first edit applying decides success on solvable tasks (72% against 14%); the synthetic bench did not predict the agent, so candidate edits are screened on the agent's own first-edit states.

## Hypothesis (stated before running)
An edit raises the share of first edits that apply on held-out real states above the unedited model and above its matched control.

## Configuration
```json
{
  "arms": [
    "combo",
    "combo_ctl",
    "ctl_L10",
    "ctrl014_a2",
    "e4b_tx",
    "esc014_a2",
    "ins_L10",
    "teacher",
    "zero",
    "instruction_text"
  ],
  "caps": [
    2048,
    512
  ],
  "sampling": {
    "temperature": 0.2,
    "top_p": 0.95
  }
}
```

## Results
```json
{
  "states": 461,
  "summary": {
    "base": {
      "counts": {
        "other_tool": 128,
        "not_found": 55,
        "escape": 14,
        "applies": 156,
        "no_call": 73,
        "multiple": 6,
        "empty": 21,
        "other_file": 8
      },
      "applies": 156,
      "applies_solvable_tasks": 30,
      "solvable_states": 56
    },
    "zero": {
      "counts": {
        "other_tool": 125,
        "not_found": 56,
        "escape": 16,
        "applies": 149,
        "empty": 21,
        "no_call": 81,
        "other_file": 6,
        "multiple": 7
      },
      "applies": 149,
      "applies_solvable_tasks": 29,
      "solvable_states": 56,
      "vs_base": [
        39,
        46,
        0.5154
      ]
    },
    "ins_L10": {
      "counts": {
        "applies": 161,
        "no_call": 75,
        "not_found": 62,
        "other_tool": 118,
        "multiple": 8,
        "other_file": 3,
        "empty": 16,
        "escape": 18
      },
      "applies": 161,
      "applies_solvable_tasks": 27,
      "solvable_states": 56,
      "vs_base": [
        48,
        43,
        0.6752
      ]
    },
    "ctl_L10": {
      "counts": {
        "applies": 156,
        "not_found": 58,
        "no_call": 76,
        "other_tool": 119,
        "escape": 21,
        "multiple": 9,
        "empty": 15,
        "other_file": 7
      },
      "applies": 156,
      "applies_solvable_tasks": 26,
      "solvable_states": 56,
      "vs_base": [
        50,
        50,
        1.0
      ]
    },
    "esc014_a2": {
      "counts": {
        "other_tool": 143,
        "not_found": 53,
        "no_call": 83,
        "empty": 53,
        "other_file": 4,
        "applies": 116,
        "escape": 5,
        "multiple": 4
      },
      "applies": 116,
      "applies_solvable_tasks": 21,
      "solvable_states": 56,
      "vs_base": [
        34,
        74,
        0.0001
      ]
    },
    "ctrl014_a2": {
      "counts": {
        "other_tool": 124,
        "not_found": 53,
        "escape": 18,
        "applies": 153,
        "other_file": 6,
        "no_call": 81,
        "empty": 19,
        "multiple": 7
      },
      "applies": 153,
      "applies_solvable_tasks": 31,
      "solvable_states": 56,
      "vs_base": [
        48,
        51,
        0.8408
      ]
    },
    "teacher": {
      "counts": {
        "applies": 117,
        "not_found": 42,
        "no_call": 65,
        "other_tool": 211,
        "empty": 13,
        "other_file": 3,
        "escape": 8,
        "multiple": 2
      },
      "applies": 117,
      "applies_solvable_tasks": 27,
      "solvable_states": 56,
      "vs_base": [
        31,
        70,
        0.0001
      ]
    },
    "e4b_tx": {
      "counts": {
        "empty": 15,
        "not_found": 74,
        "no_call": 56,
        "applies": 162,
        "other_file": 5,
        "other_tool": 120,
        "multiple": 12,
        "escape": 17
      },
      "applies": 162,
      "applies_solvable_tasks": 24,
      "solvable_states": 56,
      "vs_base": [
        54,
        48,
        0.6208
      ]
    },
    "combo": {
      "counts": {
        "applies": 94,
        "not_found": 45,
        "other_tool": 200,
        "no_call": 56,
        "empty": 42,
        "escape": 15,
        "multiple": 7,
        "other_file": 2
      },
      "applies": 94,
      "applies_solvable_tasks": 21,
      "solvable_states": 56,
      "vs_base": [
        24,
        86,
        0.0
      ]
    },
    "combo_ctl": {
      "counts": {
        "other_tool": 94,
        "no_call": 86,
        "not_found": 52,
        "applies": 180,
        "multiple": 10,
        "escape": 14,
        "other_file": 6,
        "empty": 19
      },
      "applies": 180,
      "applies_solvable_tasks": 33,
      "solvable_states": 56,
      "vs_base": [
        66,
        42,
        0.0264
      ]
    },
    "instruction_text": {
      "counts": {
        "other_tool": 122,
        "no_call": 70,
        "not_found": 53,
        "applies": 174,
        "multiple": 13,
        "other_file": 6,
        "empty": 16,
        "escape": 7
      },
      "applies": 174,
      "applies_solvable_tasks": 32,
      "solvable_states": 56,
      "vs_base": [
        59,
        41,
        0.0886
      ]
    },
    "ins_L10_vs_ctl_L10": [
      42,
      37,
      0.653
    ],
    "esc014_a2_vs_ctrl014_a2": [
      33,
      70,
      0.0003
    ],
    "combo_vs_combo_ctl": [
      23,
      109,
      0.0
    ]
  },
  "headline": "first-edit bench, edits that apply of 461: base 156, zero 149, ins_L10 161, ctl_L10 156, esc014_a2 116, ctrl014_a2 153, teacher 117, e4b_tx 162, combo 94, combo_ctl 180, instruction_text 174",
  "system_monitor": {
    "samples": 193
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

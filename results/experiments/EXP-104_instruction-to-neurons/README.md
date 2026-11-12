# EXP-104: instruction-to-neurons

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-09T12:54:58.259895Z / 2026-11-09T13:40:50.367096Z
- **Wall time:** 45.9 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-092, EXP-102
- **Command:** `experiments/scripts/exp104_instruction_to_neurons.py`

## Why this experiment
The one lab result outside the noise floor is an instruction (D-098, D-099); its effect on E2B's internal states, written into one MLP layer in closed form, would carry the behaviour in the weights.

## Hypothesis (stated before running)
A down_proj edit fitted to reproduce the instruction's state change raises applying edits on unseen bench states above the unedited arm's noise range, approaching the instruction itself, and beyond the same edit fitted to a neutral instruction.

## Configuration
```json
{
  "layers": [
    10,
    14,
    20
  ],
  "rank": 128,
  "states": 160,
  "instruction": "The bug is one small mistake on one line. Once you have read the code that contains it, fix it right away with edit_file. Copy old_string character for character from the file content you read, without adding escape characters, and in new_string change only the wrong token or operator. Do not rewrite or restructure code.",
  "control_instruction": "The repository is a Python project. Its files use four spaces for indentation and its tests are written with pytest. The tools listed above are the only ones available in this session, and every tool call returns its result as JSON. Paths are given relative to the repository root, which is also the working directory of every command you run.",
  "arms": [
    "ins_L10",
    "ins_L14",
    "ins_L20",
    "ins_L14_x2",
    "ctl_L10",
    "ctl_L14",
    "ctl_L20"
  ]
}
```

## Results
```json
{
  "fit": {
    "states": 160,
    "rows_per_layer": 46066,
    "layers": {
      "ins_L10": {
        "target_norm_mean": 3.8033,
        "fit_r2": 0.504,
        "rank_energy_share": 0.8723,
        "delta_rel_to_W": 0.3635
      },
      "ctl_L10": {
        "target_norm_mean": 2.3692,
        "fit_r2": 0.4251,
        "rank_energy_share": 0.9305,
        "delta_rel_to_W": 0.35885
      },
      "ins_L14": {
        "target_norm_mean": 2.275,
        "fit_r2": 0.6611,
        "rank_energy_share": 0.6968,
        "delta_rel_to_W": 0.19016
      },
      "ctl_L14": {
        "target_norm_mean": 1.3363,
        "fit_r2": 0.44,
        "rank_energy_share": 0.8014,
        "delta_rel_to_W": 0.15126
      },
      "ins_L20": {
        "target_norm_mean": 3.1413,
        "fit_r2": 0.7751,
        "rank_energy_share": 0.5801,
        "delta_rel_to_W": 0.27697
      },
      "ctl_L20": {
        "target_norm_mean": 1.7292,
        "fit_r2": 0.6733,
        "rank_energy_share": 0.6841,
        "delta_rel_to_W": 0.21124
      }
    }
  },
  "bench": {
    "base": {
      "applies": 28,
      "correct": 5,
      "esc": 8,
      "selfcopy": 8
    },
    "ins_L10": {
      "applies": 37,
      "correct": 4,
      "esc": 3,
      "selfcopy": 8,
      "applies_gain_loss_p": [
        17,
        8,
        0.108
      ],
      "correct_gain_loss_p": [
        3,
        4,
        1.0
      ]
    },
    "ctl_L10": {
      "applies": 28,
      "correct": 3,
      "esc": 8,
      "selfcopy": 11,
      "applies_gain_loss_p": [
        11,
        11,
        1.0
      ],
      "correct_gain_loss_p": [
        3,
        5,
        0.727
      ]
    },
    "ins_L14": {
      "applies": 29,
      "correct": 7,
      "esc": 8,
      "selfcopy": 11,
      "applies_gain_loss_p": [
        10,
        9,
        1.0
      ],
      "correct_gain_loss_p": [
        4,
        2,
        0.688
      ]
    },
    "ins_L14_x2": {
      "applies": 29,
      "correct": 8,
      "esc": 6,
      "selfcopy": 9,
      "applies_gain_loss_p": [
        13,
        12,
        1.0
      ],
      "correct_gain_loss_p": [
        6,
        3,
        0.508
      ]
    },
    "ctl_L14": {
      "applies": 30,
      "correct": 6,
      "esc": 9,
      "selfcopy": 12,
      "applies_gain_loss_p": [
        10,
        8,
        0.815
      ],
      "correct_gain_loss_p": [
        3,
        2,
        1.0
      ]
    },
    "ins_L20": {
      "applies": 30,
      "correct": 9,
      "esc": 8,
      "selfcopy": 10,
      "applies_gain_loss_p": [
        12,
        10,
        0.832
      ],
      "correct_gain_loss_p": [
        6,
        2,
        0.289
      ]
    },
    "ctl_L20": {
      "applies": 27,
      "correct": 5,
      "esc": 6,
      "selfcopy": 12,
      "applies_gain_loss_p": [
        8,
        9,
        1.0
      ],
      "correct_gain_loss_p": [
        2,
        2,
        1.0
      ]
    },
    "instruction": {
      "applies": 36,
      "correct": 7,
      "esc": 7,
      "selfcopy": 8,
      "applies_gain_loss_p": [
        15,
        7,
        0.134
      ],
      "correct_gain_loss_p": [
        5,
        3,
        0.727
      ]
    }
  },
  "headline": "instruction-to-neurons, quick bench (96 states): applies/correct base 28/5, ins_L10 37/4, ctl_L10 28/3, ins_L14 29/7, ins_L14_x2 29/8, ctl_L14 30/6, ins_L20 30/9, ctl_L20 27/5, instruction 36/7",
  "system_monitor": {
    "samples": 47
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

# EXP-120: compiled-self-instruction

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-10T04:43:17.181309Z / 2026-11-10T06:50:39.813037Z
- **Wall time:** 127.4 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-104, EXP-116, EXP-118, EXP-119
- **Command:** `experiments/scripts/exp120_compiled_self_instruction.py`

## Why this experiment
The forced self-instruction raises applying first edits by 44%; neurons cannot write it and E2B ignores it as an instruction, so its effect on E2B's states is compiled into a mid layer instead.

## Hypothesis (stated before running)
The compiled prefix raises first edits that apply on FEB above unedited and above the compiled neutral sentence.

## Configuration
```json
{
  "fits": [
    [
      "q",
      10
    ],
    [
      "q",
      14
    ],
    [
      "n",
      10
    ]
  ],
  "rank": 128,
  "continuation_tokens": 384,
  "sentences": {
    "q": "Before editing, I will copy the exact lines I am going to change from the file content above, character for character:\n```python\n",
    "n": "Before editing, I will restate in my own words what the issue asks for and which behaviour of the code is wrong.\n"
  }
}
```

## Results
```json
{
  "fit": {
    "states_used": {
      "q": 151,
      "n": 151
    },
    "fits": {
      "inj_q_L10": {
        "rows": 60601,
        "target_norm_mean": 12.3541,
        "fit_r2": 0.3486,
        "rank_energy_share": 0.6536,
        "delta_rel_to_W": 0.53415
      },
      "inj_q_L14": {
        "rows": 60601,
        "target_norm_mean": 7.6634,
        "fit_r2": 0.7075,
        "rank_energy_share": 0.5336,
        "delta_rel_to_W": 0.45551
      },
      "inj_n_L10": {
        "rows": 57366,
        "target_norm_mean": 7.7434,
        "fit_r2": 0.2975,
        "rank_energy_share": 0.7992,
        "delta_rel_to_W": 0.40628
      }
    }
  },
  "summary": {
    "base": {
      "feb_counts": {
        "applies": 141,
        "no_call": 96,
        "other_tool": 126,
        "noop": 17,
        "empty": 19,
        "not_found": 49,
        "other_file": 9,
        "multiple": 4
      },
      "applies": 141,
      "holds_reference": 19,
      "reference_states": 281,
      "quoted": 258,
      "premature_edits": 17,
      "premature_states": 128,
      "median_thought_chars": 2199,
      "applies_vs_inj_n_L10": [
        41,
        63,
        0.039
      ],
      "holds_reference_vs_inj_n_L10": [
        8,
        7,
        1.0
      ],
      "premature_edit_vs_inj_n_L10": [
        4,
        7,
        0.5488
      ]
    },
    "forced_quote": {
      "feb_counts": {
        "applies": 220,
        "no_call": 36,
        "not_found": 64,
        "other_tool": 58,
        "noop": 30,
        "multiple": 21,
        "other_file": 17,
        "empty": 15
      },
      "applies": 220,
      "holds_reference": 23,
      "reference_states": 281,
      "quoted": 461,
      "premature_edits": 38,
      "premature_states": 128,
      "median_thought_chars": 2050,
      "applies_vs_base": [
        105,
        26,
        0.0
      ],
      "holds_reference_vs_base": [
        10,
        6,
        0.4545
      ],
      "premature_edit_vs_base": [
        27,
        6,
        0.0003
      ],
      "applies_vs_inj_n_L10": [
        93,
        36,
        0.0
      ],
      "holds_reference_vs_inj_n_L10": [
        9,
        4,
        0.2668
      ],
      "premature_edit_vs_inj_n_L10": [
        25,
        7,
        0.0021
      ]
    },
    "inj_q_L10": {
      "feb_counts": {
        "other_tool": 112,
        "applies": 147,
        "noop": 24,
        "no_call": 95,
        "not_found": 49,
        "empty": 21,
        "multiple": 7,
        "other_file": 6
      },
      "applies": 147,
      "holds_reference": 21,
      "reference_states": 281,
      "quoted": 252,
      "premature_edits": 17,
      "premature_states": 128,
      "median_thought_chars": 1846,
      "applies_vs_base": [
        53,
        47,
        0.6173
      ],
      "holds_reference_vs_base": [
        8,
        6,
        0.7905
      ],
      "premature_edit_vs_base": [
        9,
        9,
        1.0
      ],
      "applies_vs_inj_n_L10": [
        39,
        55,
        0.1214
      ],
      "holds_reference_vs_inj_n_L10": [
        7,
        4,
        0.5488
      ],
      "premature_edit_vs_inj_n_L10": [
        9,
        12,
        0.6636
      ]
    },
    "inj_q_L14": {
      "feb_counts": {
        "other_tool": 94,
        "applies": 135,
        "noop": 22,
        "no_call": 138,
        "not_found": 47,
        "empty": 20,
        "multiple": 3,
        "other_file": 2
      },
      "applies": 135,
      "holds_reference": 22,
      "reference_states": 281,
      "quoted": 197,
      "premature_edits": 20,
      "premature_states": 128,
      "median_thought_chars": 3531,
      "applies_vs_base": [
        50,
        56,
        0.6274
      ],
      "holds_reference_vs_base": [
        9,
        6,
        0.6072
      ],
      "premature_edit_vs_base": [
        9,
        6,
        0.6072
      ],
      "applies_vs_inj_n_L10": [
        44,
        72,
        0.0118
      ],
      "holds_reference_vs_inj_n_L10": [
        10,
        6,
        0.4545
      ],
      "premature_edit_vs_inj_n_L10": [
        9,
        9,
        1.0
      ]
    },
    "inj_n_L10": {
      "feb_counts": {
        "applies": 163,
        "not_found": 57,
        "noop": 18,
        "other_tool": 118,
        "no_call": 74,
        "empty": 16,
        "multiple": 9,
        "other_file": 6
      },
      "applies": 163,
      "holds_reference": 18,
      "reference_states": 281,
      "quoted": 239,
      "premature_edits": 20,
      "premature_states": 128,
      "median_thought_chars": 1769,
      "applies_vs_base": [
        63,
        41,
        0.039
      ],
      "holds_reference_vs_base": [
        7,
        8,
        1.0
      ],
      "premature_edit_vs_base": [
        7,
        4,
        0.5488
      ]
    }
  },
  "headline": "compiled self-instruction (state injection) on FEB (461 states): applies / holds reference / quoted; premature edits of 128: base 141/19/258; 17, forced_quote 220/23/461; 38, inj_q_L10 147/21/252; 17, inj_q_L14 135/22/197; 20, inj_n_L10 163/18/239; 20",
  "system_monitor": {
    "samples": 128
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

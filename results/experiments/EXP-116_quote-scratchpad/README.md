# EXP-116: quote-scratchpad

- **Status:** OK
- **Category:** diagnosis
- **Started / finished (UTC):** 2026-11-10T01:46:16.281945Z / 2026-11-10T02:15:35.012766Z
- **Wall time:** 29.3 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-112
- **Command:** `experiments/scripts/exp116_quote_scratchpad.py`

## Why this experiment
E2B's thought quotes the code before 77% of first edits that apply and 44% of those that fail; whether quoting causes the edit to apply decides which neuron (or setting) to build next.

## Hypothesis (stated before running)
Forcing the thought to open with a quote of the lines to change raises first edits that apply on real states above a neutral prefix of similar length and above the recorded state.

## Configuration
```json
{
  "arms": [
    "base",
    "quote",
    "neutral",
    "fence",
    "closed"
  ],
  "caps": [
    2048,
    512
  ],
  "sampling": {
    "temperature": 0.2,
    "top_p": 0.95
  },
  "prefixes": {
    "base": "",
    "quote": "Before editing, I will copy the exact lines I am going to change from the file content above, character for character:\n```python\n",
    "neutral": "Before editing, I will restate in my own words what the issue asks for and which behaviour of the code is wrong.\n",
    "fence": "```python\n",
    "closed": null
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
        "empty": 24,
        "applies": 153,
        "other_tool": 122,
        "no_call": 78,
        "not_found": 45,
        "other_file": 8,
        "multiple": 4,
        "noop": 16,
        "escape": 11
      },
      "applies": 153,
      "holds_reference": 25,
      "reference_states": 281,
      "quoted": 270,
      "applies_solvable_tasks": 27,
      "holds_reference_solvable": 12,
      "solvable_states": 56
    },
    "quote": {
      "counts": {
        "applies": 221,
        "no_call": 46,
        "not_found": 51,
        "other_tool": 62,
        "noop": 30,
        "multiple": 9,
        "empty": 9,
        "other_file": 18,
        "escape": 15
      },
      "applies": 221,
      "holds_reference": 26,
      "reference_states": 281,
      "quoted": 461,
      "applies_solvable_tasks": 32,
      "holds_reference_solvable": 10,
      "solvable_states": 56,
      "applies_vs_base": [
        100,
        32,
        0.0
      ],
      "holds_reference_vs_base": [
        6,
        5,
        1.0
      ]
    },
    "neutral": {
      "counts": {
        "other_tool": 106,
        "applies": 159,
        "noop": 25,
        "not_found": 47,
        "other_file": 8,
        "no_call": 73,
        "empty": 25,
        "multiple": 4,
        "escape": 14
      },
      "applies": 159,
      "holds_reference": 21,
      "reference_states": 281,
      "quoted": 259,
      "applies_solvable_tasks": 29,
      "holds_reference_solvable": 8,
      "solvable_states": 56,
      "applies_vs_base": [
        53,
        47,
        0.6173
      ],
      "holds_reference_vs_base": [
        5,
        9,
        0.424
      ]
    },
    "fence": {
      "counts": {
        "other_tool": 110,
        "no_call": 110,
        "empty": 12,
        "applies": 145,
        "noop": 26,
        "not_found": 43,
        "multiple": 7,
        "escape": 5,
        "other_file": 3
      },
      "applies": 145,
      "holds_reference": 23,
      "reference_states": 281,
      "quoted": 461,
      "applies_solvable_tasks": 27,
      "holds_reference_solvable": 9,
      "solvable_states": 56,
      "applies_vs_base": [
        52,
        60,
        0.5085
      ],
      "holds_reference_vs_base": [
        5,
        7,
        0.7744
      ]
    },
    "closed": {
      "counts": {
        "applies": 115,
        "no_call": 183,
        "noop": 12,
        "other_tool": 107,
        "empty": 9,
        "not_found": 19,
        "escape": 8,
        "multiple": 4,
        "other_file": 4
      },
      "applies": 115,
      "holds_reference": 21,
      "reference_states": 281,
      "quoted": 0,
      "applies_solvable_tasks": 20,
      "holds_reference_solvable": 9,
      "solvable_states": 56,
      "applies_vs_base": [
        41,
        79,
        0.0007
      ],
      "holds_reference_vs_base": [
        4,
        8,
        0.3877
      ]
    },
    "quote_vs_neutral_applies": [
      96,
      34,
      0.0
    ],
    "quote_vs_neutral_holds_reference": [
      8,
      3,
      0.2266
    ],
    "base_edit_calls_applied_when_quoted": [
      134,
      196
    ],
    "base_edit_calls_applied_when_not_quoted": [
      19,
      65
    ]
  },
  "headline": "FEB, first edits that apply of 461 / hold the reference change of 281: base 153/25, quote 221/26, neutral 159/21, fence 145/23, closed 115/21",
  "system_monitor": {
    "samples": 30
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

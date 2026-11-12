# EXP-113: first-edit-failure-neuron

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-09T21:53:31.660097Z / 2026-11-09T23:08:48.318840Z
- **Wall time:** 75.3 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-112, EXP-080
- **Command:** `experiments/scripts/exp113_first_edit_neuron.py`

## Why this experiment
First edits that fail mostly come from editing unread code; a neuron that detects a failing first edit at the tool-name token and writes read_file over edit_file would turn those into reads.

## Hypothesis (stated before running)
On held-out tasks' first-edit states the neuron lowers failing first edits against unedited and its shuffled-label control without lowering applying edits.

## Configuration
```json
{
  "layer": 34,
  "ridge": 1.0,
  "arms": [
    "neuron",
    "neuron_half",
    "control",
    "zero"
  ]
}
```

## Results
```json
{
  "fit": {
    "dev": 230,
    "test": 231,
    "dev_failed": 109,
    "test_failed": 106,
    "neuron_auc_test": 0.6795,
    "neuron_auc_dev": 0.7776,
    "neuron_test_mean_failed": 0.531,
    "neuron_test_mean_applied": 0.432,
    "control_auc_test": 0.5741,
    "control_auc_dev": 0.6478,
    "control_test_mean_failed": 0.498,
    "control_test_mean_applied": 0.492,
    "c": 7.0,
    "median_edit_over_read_margin_failed": 6.0,
    "arms": {
      "neuron": "/content/repo/artifacts_local/EXP-113/neuron",
      "neuron_half": "/content/repo/artifacts_local/EXP-113/neuron_half",
      "control": "/content/repo/artifacts_local/EXP-113/control"
    }
  },
  "summary": {
    "base": {
      "counts": {
        "read_file": 38,
        "other_tool": 24,
        "applies": 76,
        "no_call": 50,
        "edit_failed": 41,
        "edit_other_file": 2
      }
    },
    "neuron": {
      "counts": {
        "read_file": 38,
        "edit_failed": 50,
        "no_call": 49,
        "applies": 68,
        "other_tool": 25,
        "edit_other_file": 1
      },
      "applies_vs_base": [
        16,
        24,
        0.2682
      ],
      "edit_failed_vs_base": [
        26,
        17,
        0.2221
      ]
    },
    "neuron_half": {
      "counts": {
        "applies": 81,
        "edit_failed": 37,
        "read_file": 41,
        "no_call": 49,
        "other_tool": 21,
        "edit_other_file": 2
      },
      "applies_vs_base": [
        24,
        19,
        0.5424
      ],
      "edit_failed_vs_base": [
        18,
        22,
        0.6358
      ]
    },
    "control": {
      "counts": {
        "read_file": 44,
        "edit_failed": 46,
        "applies": 72,
        "no_call": 43,
        "other_tool": 21,
        "edit_other_file": 5
      },
      "applies_vs_base": [
        17,
        21,
        0.6271
      ],
      "edit_failed_vs_base": [
        25,
        20,
        0.5515
      ]
    },
    "zero": {
      "counts": {
        "read_file": 36,
        "edit_failed": 46,
        "applies": 80,
        "no_call": 45,
        "other_tool": 21,
        "edit_other_file": 3
      },
      "applies_vs_base": [
        20,
        16,
        0.6177
      ],
      "edit_failed_vs_base": [
        21,
        16,
        0.5114
      ]
    }
  },
  "headline": "first-edit failure neuron on held-out FEB states: base applies 76 failed 41 read 38, neuron applies 68 failed 50 read 38, neuron_half applies 81 failed 37 read 41, control applies 72 failed 46 read 44, zero applies 80 failed 46 read 36",
  "system_monitor": {
    "samples": 76
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

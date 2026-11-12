# EXP-063: atlas-causal-ablation

- **Status:** OK
- **Category:** localization
- **Started / finished (UTC):** 2026-11-07T19:52:19.479846Z / 2026-11-08T01:26:51.090298Z
- **Wall time:** 334.5 min
- **Commit:** `c238d7a710f149472dfa38eeedb2bf85410e0128` (dirty=True)
- **Depends on:** EXP-062
- **Command:** `experiments/scripts/exp063_atlas_causal_ablation.py`

## Why this experiment
The atlas shows which neurons respond to which activity. Only intervention shows which are responsible for it.

## Hypothesis (stated before running)
Each group's ablation damages its own function more than matched random neurons do.

## Configuration
```json
{
  "arms": {
    "old_string_on": [
      13,
      9
    ],
    "old_string_off": [
      5,
      6
    ],
    "syntax_on": [
      3,
      12,
      1,
      4
    ],
    "syntax_off": [
      0,
      14,
      7
    ],
    "whitespace_on": [
      11
    ],
    "system_on": [
      2
    ]
  },
  "states": 32,
  "max_prompt": 12000,
  "split": "dev",
  "seed": 0
}
```

## Results
```json
{
  "arms": {
    "q4_0": {
      "base": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 5,
          "applied": 2,
          "apply_rate": 0.4,
          "not_found": 2,
          "identical": 1,
          "multiple": 0,
          "backslash_edits": 1,
          "other_tool": 0,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 22,
          "applied": 18,
          "apply_rate": 0.8182,
          "not_found": 3,
          "identical": 1,
          "multiple": 0,
          "backslash_edits": 2,
          "other_tool": 4,
          "no_tool": 1,
          "loops": 3
        },
        "all": {
          "n": 32,
          "edit_calls": 27,
          "applied": 20,
          "apply_rate": 0.7407,
          "not_found": 5,
          "identical": 2,
          "multiple": 0,
          "backslash_edits": 3,
          "other_tool": 4,
          "no_tool": 1,
          "loops": 3
        },
        "mean_tokens": 1612.3
      },
      "old_string_on": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 5,
          "applied": 3,
          "apply_rate": 0.6,
          "not_found": 1,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 0,
          "other_tool": 0,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 21,
          "applied": 14,
          "apply_rate": 0.6667,
          "not_found": 4,
          "identical": 2,
          "multiple": 0,
          "backslash_edits": 5,
          "other_tool": 6,
          "no_tool": 0,
          "loops": 1
        },
        "all": {
          "n": 32,
          "edit_calls": 26,
          "applied": 17,
          "apply_rate": 0.6538,
          "not_found": 5,
          "identical": 2,
          "multiple": 1,
          "backslash_edits": 5,
          "other_tool": 6,
          "no_tool": 0,
          "loops": 1
        },
        "mean_tokens": 1212.8
      },
      "rand_old_string_on": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 3,
          "applied": 0,
          "apply_rate": 0.0,
          "not_found": 2,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 2,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 25,
          "applied": 19,
          "apply_rate": 0.76,
          "not_found": 4,
          "identical": 1,
          "multiple": 0,
          "backslash_edits": 7,
          "other_tool": 2,
          "no_tool": 0,
          "loops": 4
        },
        "all": {
          "n": 32,
          "edit_calls": 28,
          "applied": 19,
          "apply_rate": 0.6786,
          "not_found": 6,
          "identical": 1,
          "multiple": 1,
          "backslash_edits": 8,
          "other_tool": 4,
          "no_tool": 0,
          "loops": 4
        },
        "mean_tokens": 1571.0
      },
      "old_string_off": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 4,
          "applied": 1,
          "apply_rate": 0.25,
          "not_found": 2,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 23,
          "applied": 17,
          "apply_rate": 0.7391,
          "not_found": 5,
          "identical": 0,
          "multiple": 0,
          "backslash_edits": 1,
          "other_tool": 4,
          "no_tool": 0,
          "loops": 3
        },
        "all": {
          "n": 32,
          "edit_calls": 27,
          "applied": 18,
          "apply_rate": 0.6667,
          "not_found": 7,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 2,
          "other_tool": 5,
          "no_tool": 0,
          "loops": 3
        },
        "mean_tokens": 1153.9
      },
      "syntax_on": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 4,
          "applied": 1,
          "apply_rate": 0.25,
          "not_found": 2,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 26,
          "applied": 10,
          "apply_rate": 0.3846,
          "not_found": 12,
          "identical": 2,
          "multiple": 1,
          "backslash_edits": 8,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 3
        },
        "all": {
          "n": 32,
          "edit_calls": 30,
          "applied": 11,
          "apply_rate": 0.3667,
          "not_found": 14,
          "identical": 2,
          "multiple": 2,
          "backslash_edits": 9,
          "other_tool": 2,
          "no_tool": 0,
          "loops": 3
        },
        "mean_tokens": 1448.8
      },
      "rand_syntax_on": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 4,
          "applied": 1,
          "apply_rate": 0.25,
          "not_found": 2,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 22,
          "applied": 15,
          "apply_rate": 0.6818,
          "not_found": 6,
          "identical": 0,
          "multiple": 0,
          "backslash_edits": 4,
          "other_tool": 5,
          "no_tool": 0,
          "loops": 5
        },
        "all": {
          "n": 32,
          "edit_calls": 26,
          "applied": 16,
          "apply_rate": 0.6154,
          "not_found": 8,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 5,
          "other_tool": 6,
          "no_tool": 0,
          "loops": 5
        },
        "mean_tokens": 1243.6
      },
      "syntax_off": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 2,
          "applied": 1,
          "apply_rate": 0.5,
          "not_found": 1,
          "identical": 0,
          "multiple": 0,
          "backslash_edits": 0,
          "other_tool": 3,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 19,
          "applied": 13,
          "apply_rate": 0.6842,
          "not_found": 2,
          "identical": 0,
          "multiple": 0,
          "backslash_edits": 3,
          "other_tool": 8,
          "no_tool": 0,
          "loops": 0
        },
        "all": {
          "n": 32,
          "edit_calls": 21,
          "applied": 14,
          "apply_rate": 0.6667,
          "not_found": 3,
          "identical": 0,
          "multiple": 0,
          "backslash_edits": 3,
          "other_tool": 11,
          "no_tool": 0,
          "loops": 0
        },
        "mean_tokens": 918.3
      },
      "whitespace_on": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 4,
          "applied": 1,
          "apply_rate": 0.25,
          "not_found": 2,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 25,
          "applied": 18,
          "apply_rate": 0.72,
          "not_found": 6,
          "identical": 0,
          "multiple": 0,
          "backslash_edits": 4,
          "other_tool": 2,
          "no_tool": 0,
          "loops": 3
        },
        "all": {
          "n": 32,
          "edit_calls": 29,
          "applied": 19,
          "apply_rate": 0.6552,
          "not_found": 8,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 5,
          "other_tool": 3,
          "no_tool": 0,
          "loops": 3
        },
        "mean_tokens": 1193.8
      },
      "system_on": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 4,
          "applied": 1,
          "apply_rate": 0.25,
          "not_found": 2,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 26,
          "applied": 21,
          "apply_rate": 0.8077,
          "not_found": 3,
          "identical": 1,
          "multiple": 0,
          "backslash_edits": 5,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 2
        },
        "all": {
          "n": 32,
          "edit_calls": 30,
          "applied": 22,
          "apply_rate": 0.7333,
          "not_found": 5,
          "identical": 1,
          "multiple": 1,
          "backslash_edits": 6,
          "other_tool": 2,
          "no_tool": 0,
          "loops": 2
        },
        "mean_tokens": 1256.2
      }
    },
    "f16": {
      "base": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 4,
          "applied": 1,
          "apply_rate": 0.25,
          "not_found": 2,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 24,
          "applied": 23,
          "apply_rate": 0.9583,
          "not_found": 1,
          "identical": 0,
          "multiple": 0,
          "backslash_edits": 5,
          "other_tool": 3,
          "no_tool": 0,
          "loops": 1
        },
        "all": {
          "n": 32,
          "edit_calls": 28,
          "applied": 24,
          "apply_rate": 0.8571,
          "not_found": 3,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 6,
          "other_tool": 4,
          "no_tool": 0,
          "loops": 1
        },
        "mean_tokens": 1219.8
      },
      "old_string_on": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 3,
          "applied": 1,
          "apply_rate": 0.3333,
          "not_found": 1,
          "identical": 0,
          "multiple": 1,
          "backslash_edits": 0,
          "other_tool": 2,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 26,
          "applied": 21,
          "apply_rate": 0.8077,
          "not_found": 2,
          "identical": 2,
          "multiple": 0,
          "backslash_edits": 1,
          "other_tool": 1,
          "no_tool": 0,
          "loops": 1
        },
        "all": {
          "n": 32,
          "edit_calls": 29,
          "applied": 22,
          "apply_rate": 0.7586,
          "not_found": 3,
          "identical": 2,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 3,
          "no_tool": 0,
          "loops": 1
        },
        "mean_tokens": 1140.5
      },
      "rand_old_string_on": {
        "skipped_too_long": [],
        "after_failed_edit_state": {
          "n": 5,
          "edit_calls": 5,
          "applied": 1,
          "apply_rate": 0.2,
          "not_found": 2,
          "identical": 1,
          "multiple": 1,
          "backslash_edits": 1,
          "other_tool": 0,
          "no_tool": 0,
          "loops": 0
        },
        "after_applied_edit_state": {
          "n": 27,
          "edit_calls": 23,
          "applied": 19,
          "apply_rate": 0.8261,
          "not_found": 3,
          "identical": 0,
          "multiple": 0,
          "backslash_edits": 5,
          "other_tool": 4,
          "no_tool": 0,
          "loops": 2
        },
        "all": {
          "n": 32,
          "edit_calls": 28,
          "applied": 20,
          "apply_rate": 0.7143,
          "not_found": 5,
          "identical": 1,
          "multiple": 1,
          "backslash_edits": 6,
          "other_tool": 4,
          "no_tool": 0,
          "loops": 2
        },
        "mean_tokens": 1187.3
      }
    }
  },
  "headline": "Q4_0 apply rate base 0.74, old_string_on 0.65, rand_old_string_on 0.68, old_string_off 0.67, syntax_on 0.37, rand_syntax_on 0.62, syntax_off 0.67, whitespace_on 0.66, system_on 0.73",
  "system_monitor": {
    "samples": 555,
    "ram_free_gb": {
      "min": 7.84,
      "max": 19.54,
      "mean": 11.96
    },
    "gpu_shared_gb": {
      "min": 1.13,
      "max": 7.28,
      "mean": 4.85
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 88.85
    },
    "cpu_package_w": {
      "min": 7.5,
      "max": 43.7,
      "mean": 22.7
    },
    "cpu_core_w": {
      "min": 0.7,
      "max": 19.7,
      "mean": 3.09
    },
    "igpu_w": {
      "min": 0,
      "max": 24.8,
      "mean": 12.74
    },
    "battery_pct": {
      "min": 59,
      "max": 100,
      "mean": 95.65
    },
    "cpu_pct": {
      "min": 0,
      "max": 100,
      "mean": 12.3
    },
    "cpu_perf_pct": {
      "min": 29.3,
      "max": 163.8,
      "mean": 68.73
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "passive_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "throttle_reasons": {
      "min": 0,
      "max": 0,
      "mean": 0.0
    },
    "cpu_freq_mhz": {
      "min": 1545,
      "max": 1918,
      "mean": 1808.34
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 44585,
      "mean": 2501.83
    },
    "battery_charge_mw": {
      "min": 0,
      "max": 53979,
      "mean": 6161.37
    }
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

## Reading (2026-11-08 02:00 UTC)

32 replayed edit moments from 19 dev-split tasks, one seed; "applied" is out of each arm's own edit calls.
Q4_0: unedited 20 of 27 applied (74%; old_string not found 5). Syntax group off: 11 of 30 (37%; not found 14). Matched random
neurons off: 16 of 26 (62%; not found 8). old_string group on/off: 17 of 26 (65%) and 18 of 27 (67%), against 19 of 28 (68%) for its
random control. Syntax group "off" neurons: 14 of 21, with 11 calls to other tools. Whitespace group: 19 of 29. System group: 22 of 30.
F16: unedited 24 of 28 (86%); old_string group off 22 of 29 (76%); its random control 20 of 28 (71%).
Only the syntax group passes the hypothesis: ablating it damages exact edits beyond matched random neurons (37% against 62% applied,
not-found failures 14 against 8). The old_string group's ablation is no worse than random at either precision, so the atlas label
did not identify neurons responsible for copying old_string. One seed and 32 states: the syntax result is a lead, not a settled
effect. It is consistent with EXP-075 amplifying the syntax group, whose first real-task run solved 5 of 57 (unedited 4 to 6).

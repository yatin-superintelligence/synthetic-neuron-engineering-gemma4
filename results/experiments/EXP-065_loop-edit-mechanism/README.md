# EXP-065: loop-edit-mechanism

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-06T18:14:12.501707Z / 2026-11-06T23:44:47.202819Z
- **Wall time:** 330.6 min
- **Commit:** `ce50644addecd1740af6ea2c7f00972b4f912dd3` (dirty=True)
- **Depends on:** EXP-059
- **Command:** `experiments/scripts/exp065_loop_edit_mechanism.py`

## Why this experiment
A reviewer will ask why this dose and these layers, whether the effect reverses, and how it compares with the only published Gemma loop edit. None of these had an answer.

## Hypothesis (stated before running)
Loop rate falls with dose; adding the direction raises loops, including on controls; one or two layers carry most of the effect; the prior-art edit, built on list prompts, does less on agent states.

## Configuration
```json
{
  "arms": {
    "lam_m0.5": [
      [
        14,
        15,
        19,
        23
      ],
      -0.5
    ],
    "lam_m1.5": [
      [
        14,
        15,
        19,
        23
      ],
      -1.5
    ],
    "lam_m2": [
      [
        14,
        15,
        19,
        23
      ],
      -2.0
    ],
    "lam_p0.5": [
      [
        14,
        15,
        19,
        23
      ],
      0.5
    ],
    "lam_p1": [
      [
        14,
        15,
        19,
        23
      ],
      1.0
    ],
    "only_14": [
      [
        14
      ],
      -1.0
    ],
    "only_15": [
      [
        15
      ],
      -1.0
    ],
    "only_19": [
      [
        19
      ],
      -1.0
    ],
    "only_23": [
      [
        23
      ],
      -1.0
    ]
  },
  "prior_art": "L10:3513 x -0.8, L12:3838 x 3.0",
  "seeds": [
    0
  ],
  "early_exit": true,
  "split": "dev"
}
```

## Results
```json
{
  "arms": {
    "q4_0": {
      "base": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3333,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.6667,
          15
        ],
        "tool_call_rate_on_controls": [
          0.875,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 495.1,
        "task_net_loops": {
          "rich_3470": 1,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 2,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "lam_m0.5": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2667,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7333,
          15
        ],
        "tool_call_rate_on_controls": [
          0.75,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 443.0,
        "task_net_loops": {
          "rich_3470": 1,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 0,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 1,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "lam_m1.5": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0667,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.9333,
          15
        ],
        "tool_call_rate_on_controls": [
          0.75,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 608.3,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 0,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 1,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "lam_m2": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0667,
          15
        ],
        "loop_rate_on_controls": [
          0.125,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.9333,
          15
        ],
        "tool_call_rate_on_controls": [
          0.625,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 859.7,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 1,
          "requests_7427": 1,
          "requests_7502": 0,
          "rich_2943": 0,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "lam_p0.5": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.8,
          15
        ],
        "tool_call_rate_on_controls": [
          0.75,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 429.2,
        "task_net_loops": {
          "rich_3470": 1,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 1,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 1,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "lam_p1": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2,
          15
        ],
        "loop_rate_on_controls": [
          0.125,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.8,
          15
        ],
        "tool_call_rate_on_controls": [
          0.625,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 447.5,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 1,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 1,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "only_14": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.8,
          15
        ],
        "tool_call_rate_on_controls": [
          0.625,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 379.3,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 1,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "only_15": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3333,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.8,
          15
        ],
        "tool_call_rate_on_controls": [
          0.875,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 534.3,
        "task_net_loops": {
          "rich_3470": 1,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 2,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "only_19": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.8,
          15
        ],
        "tool_call_rate_on_controls": [
          0.875,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 447.8,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 1,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 1,
          "rich_3469": 0,
          "rich_3518": 1,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "only_23": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.8667,
          15
        ],
        "tool_call_rate_on_controls": [
          0.75,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 390.3,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 1,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      },
      "prior_art_2neuron": {
        "n": 23,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2667,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          8
        ],
        "tool_call_rate_on_loop_contexts": [
          0.7333,
          15
        ],
        "tool_call_rate_on_controls": [
          0.625,
          8
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          8
        ],
        "mean_tokens": 416.4,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 2,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 0,
          "requests_6629": 0,
          "requests_6644": 0,
          "fastapi_14786": 0,
          "fastapi_9425": 0
        }
      }
    },
    "f16": {
      "base": {
        "n": 15,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3333,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.6667,
          15
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 840.1,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 2,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 1
        }
      },
      "lam_m1.5": {
        "n": 15,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          1.0,
          15
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 444.2,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 0,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 0,
          "rich_3469": 0,
          "rich_3518": 0
        }
      },
      "lam_p1": {
        "n": 15,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.2,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.8,
          15
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 465.9,
        "task_net_loops": {
          "rich_3470": 0,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 1,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 0
        }
      },
      "prior_art_2neuron": {
        "n": 15,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.3333,
          15
        ],
        "loop_rate_on_controls": [
          0.0,
          0
        ],
        "tool_call_rate_on_loop_contexts": [
          0.6667,
          15
        ],
        "tool_call_rate_on_controls": [
          0.0,
          0
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          15
        ],
        "forced_close_on_controls": [
          0.0,
          0
        ],
        "mean_tokens": 636.2,
        "task_net_loops": {
          "rich_3470": 1,
          "rich_3472": 0,
          "requests_7205": 0,
          "requests_7315": 0,
          "requests_7328": 2,
          "requests_7427": 0,
          "requests_7502": 0,
          "rich_2943": 2,
          "rich_3469": 0,
          "rich_3518": 0
        }
      }
    }
  },
  "headline": "Q4_0 loops base 0.33, lam_m0.5 0.27, lam_m1.5 0.07, lam_m2 0.07, lam_p0.5 0.20, lam_p1 0.20, only_14 0.20, only_15 0.33, only_19 0.20, only_23 0.20, prior_art_2neuron 0.27",
  "system_monitor": {
    "samples": 278,
    "ram_free_gb": {
      "min": 9.08,
      "max": 20.99,
      "mean": 14.09
    },
    "gpu_shared_gb": {
      "min": 1.25,
      "max": 6.76,
      "mean": 4.18
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 92.13
    },
    "cpu_package_w": {
      "min": 10.7,
      "max": 35.3,
      "mean": 22.07
    },
    "cpu_core_w": {
      "min": 0.7,
      "max": 14.1,
      "mean": 2.0
    },
    "igpu_w": {
      "min": 0,
      "max": 25.3,
      "mean": 13.71
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 88,
      "mean": 11.23
    },
    "cpu_perf_pct": {
      "min": 25.9,
      "max": 158.2,
      "mean": 49.33
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
      "min": 1675,
      "max": 1918,
      "mean": 1828.79
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 0,
      "mean": 0.0
    },
    "battery_charge_mw": {
      "min": 0,
      "max": 0,
      "mean": 0.0
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

## Reading
All counts are of the 15 saved moments just before a real loop, at seed 0, unless stated. Normal moments are the 8 control moments, run at Q4_0 only.

| Change to the model | Q4_0 loops | F16 loops | Normal moments with a complete tool call, Q4_0 |
|---|---|---|---|
| None | 5 | 5 | 7 |
| Remove 0.5, all four layers | 4 | not run | 6 |
| Remove 1 (from EXP-053) | 2 | 0 | 6 |
| Remove 1.5 | 1 | 0 | 6 |
| Remove 2 | 1 | not run | 5, and 1 normal moment loops |
| Add 0.5 | 3 | not run | 6 |
| Add 1 | 3 | 3 | 5, and 1 normal moment loops |
| Remove 1, layer 14 only | 3 | not run | 5 |
| Remove 1, layer 15 only | 5 | not run | 7 |
| Remove 1, layer 19 only | 3 | not run | 7 |
| Remove 1, layer 23 only | 3 | not run | 6 |
| Published 2-neuron edit | 4 | 5 | 5 |

- **Dose.** Removal is graded, with loops falling from 5 to 1 as strength rises to 1.5. Harm to normal moments appears at 2. This gives strength a reason for the first time.
- **Reversal failed.** Adding the pattern did not raise loops on these moments. It lowered them to 3 at both precisions. So this run does not show that the pattern drives looping. Both signs disturb the loop, and removal disturbs it more. Adding it does seem to harm normal moments.
- **Layers.** Layer 15 alone does nothing, and each of the others reaches 3. Four together reach 2. Differences this small are within chance. Layer 19 alone caused no loss on normal moments, so it is the one to examine further.
- **Prior art.** The 2-neuron edit does little here: 4 at Q4_0 and 5 at F16. It was built on list prompts, on a checkpoint that may differ from ours, so this is not a fair test of that paper on its own terms. It shows only that it does not transfer to these moments.
- **Weak evidence overall.** One seed, 15 moments, and loops only in 4 to 5 tasks. The held-out test, EXP-066, follows.

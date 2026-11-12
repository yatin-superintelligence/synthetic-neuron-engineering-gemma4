# EXP-071: e4b-at-e2b-loop-moments

- **Status:** OK
- **Category:** diagnosis
- **Started / finished (UTC):** 2026-11-07T09:21:36.678399Z / 2026-11-07T13:57:07.843151Z
- **Wall time:** 275.5 min
- **Commit:** `3551cdbd33bc5fcb95fd64d2ac38621e1792572b` (dirty=True)
- **Depends on:** EXP-013K, EXP-044a, EXP-066
- **Command:** `experiments/scripts/exp071_e4b_at_e2b_loop_moments.py`

## Why this experiment
E4B shows no loops on the official stack. Whether it escapes the same states decides if loops are a small-model trait, and sets the simple baseline the loop edit must be compared with.

## Hypothesis (stated before running)
E4B loops far less than E2B from the same saved moments. If its rate matches E2B's, the official difference comes from the path taken and not from the model.

## Configuration
```json
{
  "splits": [
    "dev",
    "test"
  ],
  "seeds": [
    0,
    1
  ],
  "primary": "test split, task-level sign-flip against EXP-066 base rows",
  "early_exit": true,
  "neuron_edit": null
}
```

## Results
```json
{
  "arms": {
    "q4_0": {
      "all_moments": {
        "n": 90,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0,
          78
        ],
        "loop_rate_on_controls": [
          0.0,
          12
        ],
        "tool_call_rate_on_loop_contexts": [
          1.0,
          78
        ],
        "tool_call_rate_on_controls": [
          1.0,
          12
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          78
        ],
        "forced_close_on_controls": [
          0.0,
          12
        ],
        "mean_tokens": 454.3
      },
      "dev_loops": [
        0,
        30
      ],
      "test_loops": [
        0,
        48
      ],
      "test_against_e2b": {
        "e2b_loops": [
          15,
          48
        ],
        "tasks": 14,
        "tasks_changed": 8,
        "net_loops_fewer_on_e4b": 15,
        "sign_flip_p": 0.008,
        "by_task": {
          "fastapi_11194": 2,
          "fastapi_14360": 1,
          "rich_3278": 1,
          "rich_3905": 2,
          "rich_4079": 2,
          "fastapi_12942": 2,
          "rich_2725": 1,
          "rich_3296": 4
        }
      }
    },
    "f16": {
      "all_moments": {
        "n": 90,
        "skipped_too_long": [],
        "loop_rate_on_loop_contexts": [
          0.0,
          78
        ],
        "loop_rate_on_controls": [
          0.0,
          12
        ],
        "tool_call_rate_on_loop_contexts": [
          1.0,
          78
        ],
        "tool_call_rate_on_controls": [
          1.0,
          12
        ],
        "forced_close_on_loop_contexts": [
          0.0,
          78
        ],
        "forced_close_on_controls": [
          0.0,
          12
        ],
        "mean_tokens": 483.3
      },
      "dev_loops": [
        0,
        30
      ],
      "test_loops": [
        0,
        48
      ],
      "test_against_e2b": {
        "e2b_loops": [
          14,
          48
        ],
        "tasks": 14,
        "tasks_changed": 7,
        "net_loops_fewer_on_e4b": 14,
        "sign_flip_p": 0.0162,
        "by_task": {
          "fastapi_11194": 2,
          "rich_3278": 2,
          "rich_3905": 2,
          "rich_4079": 2,
          "fastapi_12942": 2,
          "rich_2725": 1,
          "rich_3296": 3
        }
      }
    }
  },
  "headline": "E4B at E2B loop moments: Q4_0 dev [0, 30], test [0, 48]; F16 dev [0, 30], test [0, 48] (loops, replays)",
  "system_monitor": {
    "samples": 227,
    "ram_free_gb": {
      "min": 3.77,
      "max": 19.85,
      "mean": 7.33
    },
    "gpu_shared_gb": {
      "min": 1.14,
      "max": 9.57,
      "mean": 7.93
    },
    "gpu_util_pct": {
      "min": 2,
      "max": 100,
      "mean": 89.08
    },
    "cpu_package_w": {
      "min": 15.8,
      "max": 64.9,
      "mean": 32.73
    },
    "cpu_core_w": {
      "min": 0.8,
      "max": 46.1,
      "mean": 12.12
    },
    "igpu_w": {
      "min": 0,
      "max": 28.7,
      "mean": 13.98
    },
    "battery_pct": {
      "min": 44,
      "max": 98,
      "mean": 90.81
    },
    "cpu_pct": {
      "min": 0,
      "max": 80.8,
      "mean": 35.63
    },
    "cpu_perf_pct": {
      "min": 28.5,
      "max": 142.4,
      "mean": 75.99
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
      "min": 1620,
      "max": 1918,
      "mean": 1854.46
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 42587,
      "mean": 7682.21
    },
    "battery_charge_mw": {
      "min": 0,
      "max": 59596,
      "mean": 788.33
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

E4B was restarted from the 39 saved states where E2B looped in a real run (15 development, 24 test, from 24 tasks), at seeds 0 and 1,
giving 78 replays per precision, plus 12 normal states at seed 0. Prompts, thinking budget and sampling match EXP-066 exactly, and
the tokenizer files are byte-identical.

| Of the replays shown | E4B Q4_0 | E4B F16 | E2B Q4_0 no edit | E2B F16 no edit |
|---|---|---|---|---|
| Looped, test states (48 replays) | 0 | 0 | 15 | 14 |
| Looped, development states (30 replays) | 0 | 0 | not rerun here | not rerun here |
| Ended in a complete tool call, all 78 loop replays | 78 | 78 | | |
| Normal states with a complete tool call (12) | 12 | 12 | | |

By task on the test states, E4B has fewer loops than E2B in 8 of 14 tasks at Q4_0 and 7 of 14 at F16, and more in none
(sign-flip p 0.008 and 0.0162).

The zero was checked against the saved texts (`analysis/read_texts.py`), because a constant reading can mean a broken instrument.
Every E4B reply has thought text. The same detector, run on E2B's replies of the same test states, finds 15 and 14 loops, matching
EXP-066. E4B's most repeated sentence appears 7 times at Q4_0 and 6 at F16, never the 8 the rule needs, and its highest repetition
score is 0.44, against 0.86 and 0.88 for E2B. E4B also thinks for less: median 136 and 155 words, against 211 and 284. Of E4B's 180
tool calls, 179 are well formed.

So E4B does not just avoid the paths that lead to these states. Put into the exact states that trap E2B, it leaves them every time.
Loops here are a trait of the smaller model. This gives the loop edit its fair comparison, and it makes E4B a donor for a
state-transfer experiment: the E4B activations at these moments show what a non-looping state looks like.

The E4B F16 file (13.9 GB) was removed after the run, as planned.

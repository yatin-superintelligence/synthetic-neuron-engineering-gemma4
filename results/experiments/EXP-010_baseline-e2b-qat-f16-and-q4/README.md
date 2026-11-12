# EXP-010: baseline-e2b-qat-f16-and-q4

- **Status:** FAILED: KeyError: 'task_id'
- **Category:** baseline
- **Started / finished (UTC):** 2026-11-05T22:09:29.825831Z / 2026-11-06T00:36:36.142948Z
- **Wall time:** 147.1 min
- **Commit:** `fc68afbb959f11b6dafa26da723765d32afa96d0` (dirty=True)
- **Depends on:** EXP-001, EXP-002, EXP-020
- **Command:** `experiments/scripts/exp010_baseline_e2b.py`

## Why this experiment
Reference performance for every later comparison, and a failure taxonomy: training-free edits are most likely to help where failures are mechanical (format, truncation, loops) rather than missing knowledge (Takehi et al. 2026).

## Hypothesis (stated before running)
Q4_0 scores within a few points of F16. Most failures are wrong answers (logic), with a minority of mechanical failures.

## Configuration
```json
{
  "suites": [
    "mbpp_dev",
    "mbpp_test",
    "humaneval",
    "humanevalfix",
    "repopatch",
    "mmlu500",
    "behavior"
  ],
  "decoding": "greedy",
  "parallel": 8,
  "system_prompt": null,
  "max_tokens": 8192,
  "ctx_per_slot": 16384,
  "note": "rerun without truncation; truncated first attempt archived"
}
```

## Results
```json
{
  "arms": {
    "base_f16": {
      "mbpp_dev": {
        "bench": "mbpp+",
        "n": 189,
        "passed": 113,
        "pass_rate": 0.5979,
        "categories": {
          "pass": 113,
          "wrong_answer": 48,
          "wrong_or_missing_entry_point": 10,
          "timeout": 2,
          "runtime_error": 16
        },
        "mean_gen_tokens": 404.7,
        "gen": {
          "wall_s": 1528.26,
          "gen_tokens": 76480,
          "tok_per_s": 50.0
        },
        "exec_s": 35.8,
        "total_s": 1564.1
      },
      "mbpp_test": {
        "bench": "mbpp+",
        "n": 188,
        "passed": 125,
        "pass_rate": 0.6649,
        "categories": {
          "pass": 125,
          "wrong_answer": 45,
          "truncated": 2,
          "wrong_or_missing_entry_point": 9,
          "runtime_error": 6,
          "syntax_error": 1
        },
        "mean_gen_tokens": 463.0,
        "gen": {
          "wall_s": 2255.22,
          "gen_tokens": 87041,
          "tok_per_s": 38.6
        },
        "exec_s": 5.1,
        "total_s": 2260.3
      },
      "humaneval": {
        "bench": "humaneval+",
        "n": 163,
        "passed": 126,
        "pass_rate": 0.773,
        "categories": {
          "pass": 126,
          "wrong_answer": 36,
          "runtime_error": 1
        },
        "mean_gen_tokens": 392.8,
        "gen": {
          "wall_s": 1253.1,
          "gen_tokens": 64026,
          "tok_per_s": 51.1
        },
        "exec_s": 13.8,
        "total_s": 1266.9
      },
      "humanevalfix": {
        "bench": "humanevalfix",
        "n": 164,
        "passed": 145,
        "pass_rate": 0.8841,
        "categories": {
          "pass": 145,
          "wrong_answer": 18,
          "name_or_import_error": 1
        },
        "mean_gen_tokens": 706.3,
        "gen": {
          "wall_s": 2360.67,
          "gen_tokens": 115835,
          "tok_per_s": 49.1
        },
        "exec_s": 1.8,
        "total_s": 2362.5
      },
      "repopatch": {
        "bench": "repopatch",
        "n": 141,
        "passed": 49,
        "pass_rate": 0.3475,
        "format_ok_rate": 0.9787,
        "applied_rate": 0.922,
        "categories": {
          "pass": 49,
          "tests_fail": 81,
          "format_error": 3,
          "edit_not_applicable": 8
        },
        "mean_gen_tokens": 88.1,
        "gen": {
          "wall_s": 612.43,
          "gen_tokens": 12421,
          "tok_per_s": 20.3
        },
        "total_s": 621.0
      },
      "mmlu500": {
        "bench": "mmlu",
        "n": 500,
        "passed": 267,
        "pass_rate": 0.534,
        "unparsed": 47,
        "gen": {
          "wall_s": 417.63,
          "gen_tokens": 4968,
          "tok_per_s": 11.9
        }
      },
      "behavior": {
        "bench": "behavior",
        "n": 40,
        "checks_passed": 22,
        "checks_total": 22,
        "check_rate": 1.0,
        "mean_tokens": 353.5,
        "hit_limit": 0,
        "mean_distinct2": 0.879,
        "refusals_on_safe_prompts": 6,
        "gen": {
          "wall_s": 308.86,
          "gen_tokens": 14139,
          "tok_per_s": 45.8
        },
        "pass_rate": 1.0,
        "passed": 22
      }
    }
  },
  "system_monitor": {
    "samples": 129,
    "ram_free_gb": {
      "min": 1.44,
      "max": 22.23,
      "mean": 6.12
    },
    "gpu_shared_gb": {
      "min": 1.32,
      "max": 7.61,
      "mean": 7.43
    },
    "gpu_util_pct": {
      "min": 0.1,
      "max": 100.1,
      "mean": 93.33
    },
    "thermal_c": {
      "min": 27.9,
      "max": 27.9,
      "mean": 27.9
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 35.8,
      "mean": 9.36
    },
    "cpu_perf_pct": {
      "min": 33.5,
      "max": 144.7,
      "mean": 74.74
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_freq_mhz": {
      "min": 1704,
      "max": 1918,
      "mean": 1828.74
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

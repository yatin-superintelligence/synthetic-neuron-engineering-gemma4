# EXP-132b: 12b-loop-direction

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-10T22:25:19.969752Z / 2026-11-10T22:25:58.530476Z
- **Wall time:** 0.6 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-014d, EXP-040, EXP-053
- **Command:** `/content/repo/experiments/scripts/exp132_12b_loop_direction.py`

## Why this experiment
26 of 12B's 40 runs on the reference-resolvable tasks end in a context overflow that follows a thinking loop; where it does not overflow 12B solves 10 of 14. The loop direction found and removed in E2B is rebuilt for 12B.

## Hypothesis (stated before running)
A few 12B layers separate loop tokens on held-out tasks (d' well above random directions); the agent test (EXP-133) measures loops, context overflows and tasks solved.

## Configuration
```json
{
  "model": "google/gemma-4-12B-it-qat-q4_0-unquantized",
  "ctx_tokens": 3000,
  "resp_tokens": 1500,
  "tokens_per_class": 200,
  "layers_edited": 4,
  "fit_half_seed": 132,
  "traces": "EXP-014d",
  "fixed_layers": [
    23,
    24,
    25,
    26
  ]
}
```

## Results
```json
{
  "layers": [
    23,
    24,
    25,
    26
  ],
  "top": [
    {
      "layer": 23,
      "d_prime_heldout": 1.054,
      "random_abs_d_max": 0.481,
      "random_abs_d_mean": 0.183,
      "cosine_between_halves": 0.897,
      "n_loop_tokens": 4400,
      "n_other_tokens": 6812
    },
    {
      "layer": 24,
      "d_prime_heldout": 1.041,
      "random_abs_d_max": 0.331,
      "random_abs_d_mean": 0.124,
      "cosine_between_halves": 0.907,
      "n_loop_tokens": 4400,
      "n_other_tokens": 6812
    },
    {
      "layer": 25,
      "d_prime_heldout": 1.053,
      "random_abs_d_max": 0.402,
      "random_abs_d_mean": 0.145,
      "cosine_between_halves": 0.866,
      "n_loop_tokens": 4400,
      "n_other_tokens": 6812
    },
    {
      "layer": 26,
      "d_prime_heldout": 1.069,
      "random_abs_d_max": 0.648,
      "random_abs_d_mean": 0.224,
      "cosine_between_halves": 0.928,
      "n_loop_tokens": 4400,
      "n_other_tokens": 6812
    }
  ],
  "random_overlap_with_loop_direction": {
    "23": -0.017,
    "24": 0.0052,
    "25": -0.0091,
    "26": 0.0382
  },
  "residual_write_along_direction_after_edit": {
    "23": 2e-06,
    "24": 3e-06,
    "25": 3e-06,
    "26": 3e-06
  },
  "cases_used": 47,
  "adapters": [
    "loop_edit",
    "loop_control"
  ],
  "headline": "12B loop direction: layers [23, 24, 25, 26], held-out d' L23 1.054 (random max 0.481), L24 1.041 (random max 0.331), L25 1.053 (random max 0.402), L26 1.069 (random max 0.648)",
  "system_monitor": {
    "samples": 1
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

# EXP-132: 12b-loop-direction

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-10T22:19:50.779206Z / 2026-11-10T22:21:34.309565Z
- **Wall time:** 1.7 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-014d, EXP-040, EXP-053
- **Command:** `experiments/scripts/exp132_12b_loop_direction.py`

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
  "traces": "EXP-014d"
}
```

## Results
```json
{
  "layers": [
    33,
    38,
    39,
    40
  ],
  "top": [
    {
      "layer": 40,
      "d_prime_heldout": 1.097,
      "random_abs_d_max": 0.872,
      "random_abs_d_mean": 0.428,
      "cosine_between_halves": 0.982,
      "n_loop_tokens": 4400,
      "n_other_tokens": 6812
    },
    {
      "layer": 33,
      "d_prime_heldout": 1.093,
      "random_abs_d_max": 0.808,
      "random_abs_d_mean": 0.425,
      "cosine_between_halves": 0.982,
      "n_loop_tokens": 4400,
      "n_other_tokens": 6812
    },
    {
      "layer": 38,
      "d_prime_heldout": 1.084,
      "random_abs_d_max": 0.743,
      "random_abs_d_mean": 0.365,
      "cosine_between_halves": 0.984,
      "n_loop_tokens": 4400,
      "n_other_tokens": 6812
    },
    {
      "layer": 39,
      "d_prime_heldout": 1.076,
      "random_abs_d_max": 0.955,
      "random_abs_d_mean": 0.425,
      "cosine_between_halves": 0.98,
      "n_loop_tokens": 4400,
      "n_other_tokens": 6812
    }
  ],
  "random_overlap_with_loop_direction": {
    "33": -0.0202,
    "38": -0.0008,
    "39": 0.0152,
    "40": 0.0074
  },
  "residual_write_along_direction_after_edit": {
    "33": 4e-06,
    "38": 4e-06,
    "39": 4e-06,
    "40": 2e-06
  },
  "cases_used": 47,
  "adapters": [
    "loop_edit",
    "loop_control"
  ],
  "headline": "12B loop direction: layers [33, 38, 39, 40], held-out d' L40 1.097 (random max 0.872), L33 1.093 (random max 0.808), L38 1.084 (random max 0.743), L39 1.076 (random max 0.955)",
  "system_monitor": {
    "samples": 2
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

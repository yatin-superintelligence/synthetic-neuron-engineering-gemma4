# EXP-040: loop-direction-discovery

- **Status:** OK
- **Category:** localization
- **Started / finished (UTC):** 2026-11-06T08:38:10.476226Z / 2026-11-06T08:50:11.918272Z
- **Wall time:** 12.0 min
- **Commit:** `89e98fedcb764dc84b5e176342b0d4b51ec35bc7` (dirty=True)
- **Depends on:** EXP-038
- **Command:** `experiments/scripts/exp040_loop_direction.py`

## Why this experiment
If looping is carried by a specific direction written by a few MLP layers, as Takehi et al. found for Qwen, it can be removed by a rank-1 edit that compiles exactly to a LoRA.

## Hypothesis (stated before running)
A few layers show held-out d' above 1 and well above random directions; layer ranking is stable between task halves.

## Configuration
```json
{
  "ctx_tokens": 3000,
  "resp_tokens": 1500,
  "tokens_per_class": 200,
  "split": "dev",
  "fit_half_seed": 40,
  "classes": [
    "LOOP",
    "PRE",
    "NORMAL"
  ]
}
```

## Results
```json
{
  "top_layers": [
    {
      "layer": 14,
      "d_prime_heldout": 5.319,
      "d_prime_fit_gap": 1.8586,
      "random_abs_d_max": 1.662,
      "random_abs_d_mean": 0.798,
      "cosine_between_halves": 0.958,
      "n_loop_tokens": 3000,
      "n_other_tokens": 5246
    },
    {
      "layer": 15,
      "d_prime_heldout": 4.051,
      "d_prime_fit_gap": 3.7285,
      "random_abs_d_max": 2.786,
      "random_abs_d_mean": 0.765,
      "cosine_between_halves": 0.957,
      "n_loop_tokens": 3000,
      "n_other_tokens": 5246
    },
    {
      "layer": 19,
      "d_prime_heldout": 3.775,
      "d_prime_fit_gap": 1.6513,
      "random_abs_d_max": 2.076,
      "random_abs_d_mean": 0.78,
      "cosine_between_halves": 0.953,
      "n_loop_tokens": 3000,
      "n_other_tokens": 5246
    },
    {
      "layer": 23,
      "d_prime_heldout": 3.653,
      "d_prime_fit_gap": 3.5575,
      "random_abs_d_max": 2.043,
      "random_abs_d_mean": 0.869,
      "cosine_between_halves": 0.98,
      "n_loop_tokens": 3000,
      "n_other_tokens": 5246
    },
    {
      "layer": 20,
      "d_prime_heldout": 3.384,
      "d_prime_fit_gap": 2.6375,
      "random_abs_d_max": 2.728,
      "random_abs_d_mean": 0.844,
      "cosine_between_halves": 0.979,
      "n_loop_tokens": 3000,
      "n_other_tokens": 5246
    },
    {
      "layer": 22,
      "d_prime_heldout": 3.309,
      "d_prime_fit_gap": 1.5894,
      "random_abs_d_max": 1.916,
      "random_abs_d_mean": 0.769,
      "cosine_between_halves": 0.973,
      "n_loop_tokens": 3000,
      "n_other_tokens": 5246
    }
  ],
  "headline": "top held-out d': L14 5.319 (cos 0.958), L15 4.051 (cos 0.957), L19 3.775 (cos 0.953), L23 3.653 (cos 0.98); random max 2.79",
  "system_monitor": {
    "samples": 11,
    "ram_free_gb": {
      "min": 5.56,
      "max": 19.97,
      "mean": 7.36
    },
    "gpu_shared_gb": {
      "min": 1.4,
      "max": 8.56,
      "mean": 7.63
    },
    "gpu_util_pct": {
      "min": 0.2,
      "max": 156.1,
      "mean": 85.9
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
      "max": 21.4,
      "mean": 7.75
    },
    "cpu_perf_pct": {
      "min": 120.4,
      "max": 164.3,
      "mean": 143.91
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_freq_mhz": {
      "min": 1689,
      "max": 1876,
      "mean": 1768.45
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
- **There is a loop signal, and it peaks mid-network.** Held-out d' is 5.3 at layer 14 and 4.1 at layer 15, against random
  maxima of 1.7 and 2.8. Layers 16 to 28 sit near 3. Fitted directions are stable between task halves (cosine 0.95 to 0.99).
- **The hypothesis of a few layers is only partly right.** Layer 14 stands out, but most layers from 13 to 28 separate the
  classes, and random directions separate them too, up to 2.8. So looping shifts the state broadly, on top of any specific
  direction. Tokens in a loop repeat and are not independent, so these d' values overstate certainty.
- **Layer 14 is the last layer before key-value sharing and the double-wide MLP begin at layer 15.** This may be coincidence.
- **This is description, not cause.** LOOP tokens are also late in a response and repeat earlier text, so any of these can drive
  the direction. Only removing the direction and watching behaviour can show cause.
- **The last layer is weak for a mean-difference direction:** d' 2.3 against a random maximum of 2.2. EXP-051's read-out uses
  12288 neurons and may do better, but this lowers the prior for a last-layer loop breaker.
- **Memory.** This run peaked at 8.6 GB of shared GPU memory with 5.6 GB of RAM free, a safe margin. The cause of the 1.2 GB
  low points in the stopped run and in the preflight is not established; logits_to_keep alone did not remove it.

# EXP-109e: transplant-e4b-total-effect

- **Status:** FAILED: FileNotFoundError: [Errno 2] No such file or directory: '/content/EXP-109e_fit.json'
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-09T18:57:15.491782Z / 2026-11-09T19:11:44.513035Z
- **Wall time:** 14.5 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-102b, EXP-108
- **Command:** `/content/repo/experiments/scripts/exp109_transplant_total_effect.py`

## Why this experiment
The 31B knows far more at the fix step than E4B, but its only token-tested transplant used the vocabulary map. The state map worked for E4B, and the direct-logit choice is blind to what mid-layer neurons do downstream.

## Hypothesis (stated before running)
31B neurons through the state map, chosen by total effect, raise held reference-edit token accuracy above the unedited model, above random 31B neurons and above the direct-logit choice.

## Configuration
```json
{
  "e2b_layers": [
    8,
    12,
    16,
    20,
    24,
    28,
    32
  ],
  "arms": [
    "st_dla",
    "st_tot",
    "st_rnd"
  ],
  "eval_states": 160,
  "neurons_per_layer": 128,
  "donor": "google/gemma-4-E4B-it-qat-q4_0-unquantized"
}
```

## Results
```json
{
  "system_monitor": {
    "samples": 15
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

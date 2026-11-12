# EXP-081: reread-full-context

- **Status:** OK
- **Category:** diagnostic
- **Started / finished (UTC):** 2026-11-08T00:16:54.943861Z / 2026-11-08T00:30:03.551648Z
- **Wall time:** 13.1 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-076
- **Command:** `experiments/scripts/exp081_reread_full_context.py`

## Why this experiment
EXP-076a showed no change in the re-read decision on real tasks; the read-out was fitted on 3,000-token contexts.

## Hypothesis (stated before running)
At full context the N_reread read-out on its positives is far lower than at 3,000 tokens (below 0.35).

## Configuration
```json
{
  "layer": 34,
  "ctx_exp076": 3000,
  "max_prompt": 22000,
  "negatives": 300,
  "seed": 81
}
```

## Results
```json
{
  "last3000": {
    "separation": {
      "pos_mean": 0.724,
      "pos_min": 0.158,
      "neg_median": 0.1833,
      "neg_p99": 0.8263,
      "neg_p999": 0.8663,
      "neg_max": 0.867,
      "n_pos": 163,
      "n_neg": 300
    },
    "tokens_median": 3001,
    "gap_read_minus_edit_median": -1.375,
    "flip_share_if_shift_is_3_times_readout": 0.503,
    "read_preferred_already": 0.436
  },
  "full": {
    "separation": {
      "pos_mean": 0.605,
      "pos_min": -0.153,
      "neg_median": 0.2492,
      "neg_p99": 0.7895,
      "neg_p999": 0.8349,
      "neg_max": 0.8457,
      "n_pos": 163,
      "n_neg": 300
    },
    "tokens_median": 14301,
    "gap_read_minus_edit_median": 1.312,
    "flip_share_if_shift_is_3_times_readout": 0.675,
    "read_preferred_already": 0.607
  },
  "headline": "N_reread positives: read-out 0.724 at 3,000 tokens, 0.605 at full context; predicted flip share 0.503 vs 0.675",
  "system_monitor": {
    "samples": 14
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

## Reading (2026-11-08 00:35 UTC)

All 163 re-read moments (both splits) and 300 other tool-call moments, bf16 reference E2B, layer 34.

| | last 3,000 tokens (as EXP-076 fitted) | full context (cut to 22,000; median 14,301 tokens) |
|---|---|---|
| read-out at re-read moments, mean | 0.724 | 0.605 |
| read-out elsewhere, median / 99th percentile | 0.183 / 0.826 | 0.249 / 0.790 |
| logit "read" minus "edit" at re-read moments, median | -1.375 | +1.312 |
| re-read moments where "read" already beats "edit" | 43.6% | 60.7% |
| predicted share where "read" beats "edit" with the +3 write | 50.3% | 67.5% |

The rule fixed before the run (a full-context read-out below 0.35 would mean the context mismatch explains EXP-076a) does not hold:
the read-out keeps most of its strength (0.605). The weak real-task result has a different cause: at full context the model already
prefers re-reading at 61% of these moments, and a +3 write scaled by a read-out of about 0.6 turns only about 7 more points, which is
the size of one run's noise (EXP-076a 45% against EXP-076b 31% and about 37% unedited). EXP-076's remaining seeds were stopped, and
the neuron is refitted at full context with a write strength set from data in EXP-080.

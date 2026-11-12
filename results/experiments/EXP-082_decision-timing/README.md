# EXP-082: decision-timing

- **Status:** OK
- **Category:** interpretability
- **Started / finished (UTC):** 2026-11-08T01:54:39.532390Z / 2026-11-08T02:28:58.105435Z
- **Wall time:** 34.3 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-079, EXP-080
- **Command:** `experiments/scripts/exp082_decision_timing.py`

## Why this experiment
EXP-080's neurons write at the tool-name token, where E2B's choice is already settled; find where the choice forms.

## Hypothesis (stated before running)
E2B's own action is far more readable at the tool-name position than at the first thinking token (at least 20 points).

## Configuration
```json
{
  "layer_mlp": 34,
  "layer_res": 17,
  "positions": [
    "start",
    "q25",
    "q50",
    "q75",
    "call"
  ],
  "tools": [
    "read_file",
    "edit_file",
    "search_similar_code",
    "run_command",
    "submit_patch",
    "write_file"
  ],
  "ridge": 1.0,
  "max_prompt": 22000
}
```

## Results
```json
{
  "e2b": {
    "majority_class_rate": 0.36,
    "n_test": 791,
    "mlp": {
      "start": 0.55,
      "q25": 0.422,
      "q50": 0.42,
      "q75": 0.503,
      "call": 0.973
    },
    "res": {
      "start": 0.595,
      "q25": 0.571,
      "q50": 0.552,
      "q75": 0.613,
      "call": 0.953
    },
    "logit_lens_top_tool_matches": {
      "start": 0.234,
      "q25": 0.259,
      "q50": 0.286,
      "q75": 0.315,
      "call": 0.984
    }
  },
  "t_major": {
    "majority_class_rate": 0.442,
    "n_test": 771,
    "mlp": {
      "start": 0.427,
      "q25": 0.385,
      "q50": 0.374,
      "q75": 0.357,
      "call": 0.388
    },
    "res": {
      "start": 0.409,
      "q25": 0.418,
      "q50": 0.397,
      "q75": 0.405,
      "call": 0.388
    }
  },
  "think_tokens_median": 244,
  "headline": "E2B's own action from layer-34 MLP activations: start 0.55, q25 0.422, q50 0.42, q75 0.503, call 0.973 (majority class 0.36); 31B's action: start 0.427, q25 0.385, q50 0.374, q75 0.357, call 0.388",
  "system_monitor": {
    "samples": 35
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

## Reading (2026-11-08 02:32 UTC)

Colab A100, 34 minutes. 1,542 states (six tools), probes fitted on 751 dev-split states, scored on 791 test-split states.

E2B's own next action (always guessing the most common: 36%):

| position | layer-34 MLP probe | layer-17 residual probe | model's own top tool token (of the six) |
|---|---|---|---|
| first thinking token | 55.0% | 59.5% | 23.4% |
| 25% of the thinking | 42.2% | 57.1% | 25.9% |
| 50% | 42.0% | 55.2% | 28.6% |
| 75% | 50.3% | 61.3% | 31.5% |
| tool-name token | 97.3% | 95.3% | 98.4% |

31B's majority action at the same states (always guessing the most common: 44.2%, 771 test states): probes 35.7% to 42.7% at every
position and layer, at or below that rate.

The prediction holds: E2B's action is 42 points more readable at the tool-name token than at the first thinking token, and it is
only partly readable before the thinking (55% to 60%). 31B's choice is not linearly readable from E2B's activations at any position.
Consequence for EXP-080: its read-outs cannot carry per-state knowledge of what 31B would do; they mostly detect "E2B is about to call
X" in some of X's contexts, and the write moves part of those calls to 31B's preferred Y. The EXP-080 neurons are therefore best
described as action-level policy shifts distilled from 31B's systematic preferences, with partial state selectivity (predicted flips
82% at their positives against 39% at states where 31B agreed with E2B, for search -> run_command). The control result (EXP-080b)
shows that this selectivity matters for the behavioural change.

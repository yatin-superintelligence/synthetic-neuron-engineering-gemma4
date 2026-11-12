# EXP-076: synthetic-decision-neurons

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-07T23:26:55.715761Z / 2026-11-07T23:28:59.521867Z
- **Wall time:** 2.1 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-044a
- **Command:** `experiments/scripts/exp076_synthetic_decision_neurons.py`

## Why this experiment
Add new neurons, not only edit existing ones: two read-out neurons built from official traces that act at one agent decision each.

## Hypothesis (stated before running)
Each read-out separates its decision on held-out tasks (test split) far above its control. Real-task effects are measured in the follow-up runs.

## Configuration
```json
{
  "layer": 34,
  "logit_shift": 3.0,
  "ridge": 1.0,
  "ctx": 3000,
  "runs": [
    "exp029",
    "exp037",
    "exp-044a"
  ],
  "fit_split": "dev",
  "check_split": "test"
}
```

## Results
```json
{
  "neurons": {
    "reread": {
      "n_pos": 74,
      "n_neg": 431,
      "sep_fit": {
        "pos_mean": 0.761,
        "pos_min": 0.282,
        "neg_median": 0.193,
        "neg_p99": 0.7999,
        "neg_p999": 0.849,
        "neg_max": 0.8652,
        "n_pos": 74,
        "n_neg": 431
      },
      "sep_test": {
        "pos_mean": 0.695,
        "pos_min": 0.155,
        "neg_median": 0.1859,
        "neg_p99": 0.8143,
        "neg_p999": 0.8699,
        "neg_max": 0.8721,
        "n_pos": 89,
        "n_neg": 426
      },
      "control_on_true_positives": 0.4582,
      "rms_y": 0.2562115788459778,
      "rms_h": 0.6328922510147095,
      "base_gap_at_positives": -0.312
    },
    "test": {
      "n_pos": 140,
      "n_neg": 365,
      "sep_fit": {
        "pos_mean": 0.754,
        "pos_min": 0.294,
        "neg_median": 0.1911,
        "neg_p99": 0.6744,
        "neg_p999": 0.7378,
        "neg_max": 0.7443,
        "n_pos": 140,
        "n_neg": 365
      },
      "sep_test": {
        "pos_mean": 0.639,
        "pos_min": 0.216,
        "neg_median": 0.2014,
        "neg_p99": 0.691,
        "neg_p999": 0.7682,
        "neg_max": 0.7817,
        "n_pos": 138,
        "n_neg": 377
      },
      "control_on_true_positives": 0.4918,
      "rms_y": 0.23443200439214706,
      "rms_h": 0.6458347141742706,
      "base_gap_at_positives": 8.938
    }
  },
  "headline": "reread: held-out read-out on positives 0.695, negatives p99 0.8143, control on positives 0.4582; test: held-out read-out on positives 0.639, negatives p99 0.691, control on positives 0.4918",
  "system_monitor": {
    "samples": 3
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

## Reading (2026-11-07 23:35 UTC)

Two builds with the same rows (sampling seed 76): build 1 on a Colab T4 (1.86 h; its adapters were lost to a packing bug, its numbers
are in experiments/_system/colab.log) and build 2 on a Colab G4 (124 s, this record). The held-out numbers agree to the third decimal
(N_reread positives 0.695 in both, negatives p99 0.8143 in both), so the build is reproducible across GPUs.

- **N_reread** (after a failed edit_file). Fitted on 74 positives and 431 negatives from dev-split tasks. On test-split tasks, which the
  fit never saw: read-out mean 0.695 on 89 positives (minimum 0.155) against median 0.186 on 426 negatives; 1% of negatives read above
  0.814, so the neuron is selective but not clean. The control read-out (fitted to as many random tool-name positions) gives 0.458 on
  the same positives, so the control is a broad read bias of about two thirds the strength at these moments, not a zero; it tests
  whether the selectivity matters. At the positives the unedited model prefers "edit" over "read" by 0.31 logits (median), so the
  +3 write can flip the choice.
- **N_test** (edits applied, no command run since). Held-out positives 0.639 against negatives median 0.201, p99 0.691. It cannot work
  as designed: at its positives the unedited model already prefers "run" over "submit" by 8.9 logits, so +3 changes nothing, and 47
  of 58 EXP-044a tasks end on a step with no tool call, a choice made before the tool-name position this neuron writes to. The
  premise was also wrong: E2B already runs pytest after its last applied edit in 32 of 50 tasks with an applied edit (EXP-044a). Not
  run on real tasks. The stopping decision needs its own position and write (planned as EXP-078).

## Real-task test (fixed before any result)

Run list experiments/colab_configs/batch_exp076.json, official E2B W4A16 on a Colab L4, patches scored on Kaggle by the official
verification: neuron_reread and control_reread at seeds 1, 2 and 3, and the unedited model at seed 1 on the same hardware (EXP-044d;
EXP-044b and EXP-044c are seeds 2 and 3). Primary outcome: tasks resolved of 58, neuron against control and against unedited, task-level
sign-flip test across seeds. Mechanism outcome (tools/decision_metrics.py): after a failed edit, the share of next calls that are
read_file. Unedited E2B: 21 of 50 (EXP-044a), 19 of 41 (exp029), 24 of 72 (exp037). If this share does not rise against control, the
neuron did not act on real tasks, whatever the solved count does.

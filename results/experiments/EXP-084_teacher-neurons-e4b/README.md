# EXP-084: teacher-neurons-e4b

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T05:52:33.172612Z / 2026-11-08T06:35:10.079926Z
- **Wall time:** 42.6 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-083, EXP-080
- **Command:** `experiments/scripts/exp084_teacher_neurons_e4b.py`

## Why this experiment
EXP-080 for E4B: neurons that move E4B towards what Gemma 4 31B does at the same agent state, from EXP-083's labels.

## Hypothesis (stated before running)
Pairs (E2B action X, 31B action Y) with enough confident disagreements give read-outs that separate their states on held-out tasks far above control.

## Configuration
```json
{
  "layer": 41,
  "logit_shift": "calibrated, 3 to 10",
  "ridge": 1.0,
  "min_dev_positives": 25,
  "teacher_min_votes": 3,
  "max_prompt": 22000
}
```

## Results
```json
{
  "neurons": {
    "read_file_to_run_command": {
      "x": "read_file",
      "y": "run_command",
      "c": 10,
      "n_pos": 106,
      "n_neg": 701,
      "sep_fit": {
        "pos_mean": 0.758,
        "pos_min": 0.356,
        "neg_median": 0.0671,
        "neg_p99": 0.8618,
        "neg_p999": 0.9144,
        "neg_max": 0.955,
        "n_pos": 106,
        "n_neg": 701
      },
      "sep_test": {
        "pos_mean": 0.755,
        "pos_min": 0.268,
        "neg_median": 0.0637,
        "neg_p99": 0.8764,
        "neg_p999": 0.9449,
        "neg_max": 0.9531,
        "n_pos": 104,
        "n_neg": 728
      },
      "control_on_test_positives": 0.4926,
      "dev_gap_median": -8.469,
      "predicted": {
        "test_positives": {
          "n": 104,
          "y_beats_x_before": 0.01,
          "after": 0.519
        },
        "test_agreeing_x_states": {
          "n": 150,
          "y_beats_x_before": 0.0,
          "after": 0.227
        }
      }
    },
    "search_similar_code_to_run_command": {
      "x": "search_similar_code",
      "y": "run_command",
      "c": 10,
      "n_pos": 43,
      "n_neg": 764,
      "sep_fit": {
        "pos_mean": 0.887,
        "pos_min": 0.456,
        "neg_median": 0.0229,
        "neg_p99": 0.94,
        "neg_p999": 1.0318,
        "neg_max": 1.0791,
        "n_pos": 43,
        "n_neg": 764
      },
      "sep_test": {
        "pos_mean": 0.908,
        "pos_min": 0.439,
        "neg_median": 0.0271,
        "neg_p99": 1.0061,
        "neg_p999": 1.0613,
        "neg_max": 1.0676,
        "n_pos": 53,
        "n_neg": 779
      },
      "control_on_test_positives": 0.5067,
      "dev_gap_median": -4.75,
      "predicted": {
        "test_positives": {
          "n": 53,
          "y_beats_x_before": 0.208,
          "after": 0.943
        },
        "test_agreeing_x_states": {
          "n": 18,
          "y_beats_x_before": 0.056,
          "after": 0.833
        }
      }
    },
    "edit_file_to_read_file": {
      "x": "edit_file",
      "y": "read_file",
      "c": 10,
      "n_pos": 32,
      "n_neg": 775,
      "sep_fit": {
        "pos_mean": 0.905,
        "pos_min": 0.842,
        "neg_median": 0.0084,
        "neg_p99": 0.9207,
        "neg_p999": 0.9467,
        "neg_max": 0.9494,
        "n_pos": 32,
        "n_neg": 775
      },
      "sep_test": {
        "pos_mean": 0.885,
        "pos_min": 0.783,
        "neg_median": 0.0093,
        "neg_p99": 0.921,
        "neg_p999": 0.959,
        "neg_max": 0.9666,
        "n_pos": 32,
        "n_neg": 800
      },
      "control_on_test_positives": 0.5035,
      "dev_gap_median": -6.25,
      "predicted": {
        "test_positives": {
          "n": 32,
          "y_beats_x_before": 0.0,
          "after": 0.875
        },
        "test_agreeing_x_states": {
          "n": 32,
          "y_beats_x_before": 0.0,
          "after": 0.656
        }
      }
    },
    "reread_full": {
      "x": "edit_file",
      "y": "read_file",
      "c": 10,
      "n_pos": 38,
      "n_neg": 769,
      "sep_fit": {
        "pos_mean": 0.722,
        "pos_min": 0.484,
        "neg_median": 0.28,
        "neg_p99": 0.7337,
        "neg_p999": 0.8317,
        "neg_max": 0.8824,
        "n_pos": 38,
        "n_neg": 769
      },
      "sep_test": {
        "pos_mean": 0.665,
        "pos_min": 0.006,
        "neg_median": 0.259,
        "neg_p99": 0.7566,
        "neg_p999": 0.8379,
        "neg_max": 0.8476,
        "n_pos": 39,
        "n_neg": 793
      },
      "control_on_test_positives": 0.4472,
      "dev_gap_median": 4.469,
      "predicted": {
        "test_positives": {
          "n": 39,
          "y_beats_x_before": 0.744,
          "after": 0.846
        },
        "test_agreeing_x_states": {
          "n": 29,
          "y_beats_x_before": 0.0,
          "after": 0.207
        }
      }
    }
  },
  "reread_at_full_context": null,
  "headline": "4 teacher neurons built; read_file_to_run_command (c=10): test positives 0.755 vs negatives p99 0.8764, control 0.4926, predicted {'test_positives': {'n': 104, 'y_beats_x_before': 0.01, 'after': 0.519}, 'test_agreeing_x_states': {'n': 150, 'y_beats_x_before': 0.0, 'after': 0.227}}; search_similar_code_to_run_command (c=10): test positives 0.908 vs negatives p99 1.0061, control 0.5067, predicted {'test_positives': {'n': 53, 'y_beats_x_before': 0.208, 'after': 0.943}, 'test_agreeing_x_states': {'n': 18, 'y_beats_x_before': 0.056, 'after': 0.833}}; edit_file_to_read_file (c=10): test positives 0.885 vs negatives p99 0.921, control 0.5035, predicted {'test_positives': {'n': 32, 'y_beats_x_before': 0.0, 'after': 0.875}, 'test_agreeing_x_states': {'n': 32, 'y_beats_x_before': 0.0, 'after': 0.656}}; reread_full (c=10): test positives 0.665 vs negatives p99 0.7566, control 0.4472, predicted {'test_positives': {'n': 39, 'y_beats_x_before': 0.744, 'after': 0.846}, 'test_agreeing_x_states': {'n': 29, 'y_beats_x_before': 0.0, 'after': 0.207}}",
  "system_monitor": {
    "samples": 43
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

## Reading (2026-11-08 06:45 UTC)

Colab G4, 0.74 h (the build was started on a G4 before D-050; its real-task runs are on L4s). Neurons with at least 25 dev positives
(test split: read-out on positives / negatives median / p99; control neuron on positives; dev logit gap; predicted flips before ->
after on test positives; predicted flips at test states where 31B agreed with E4B):
- search_similar_code -> run_command: 0.908 / 0.027 / 1.006; control 0.507; gap -4.75; 21% -> 94% (53); agreeing 6% -> 83% (18).
- edit_file -> read_file: 0.885 / 0.009 / 0.921; control 0.504; gap -6.25; 0% -> 88% (32); agreeing 0% -> 66% (32).
- read_file -> run_command: 0.755 / 0.064 / 0.876; control 0.493; gap -8.47; 1% -> 52% (104); agreeing 0% -> 23% (150).
- re-read after a failed edit (refitted): 0.665 / 0.259 / 0.757; gap +4.47; 74% -> 85% (39); agreeing 0% -> 21% (29).
teacher_all holds the three teacher neurons (edit_file -> run_command had too few positives for E4B). E4B's choices are less settled
at the tool-name token than E2B's (gaps 5 to 8 logits against 7 to 12), so larger shifts are predicted, with more change at
states where 31B agreed. Real-task test: experiments/colab_configs/batch_exp084.json (seed 1: teacher_all, control_all, unedited
E4B on an L4), seeds 2 and 3 in batch_exp084_seeds23.json, run only if seed 1's mechanism outcome moves.

## First real-task run (2026-11-08 08:05 UTC)

EXP-084a, teacher_all on E4B, seed 1, Colab L4; official score pending. Against the six unedited E4B Colab runs (EXP-073a to f):
read before edit 75 of 119 (63%; unedited 59% to 67%), the pre-registered primary outcome, inside the range; after a failed edit,
next call read_file 20 of 35 (57%; unedited 29% to 51%); blind retries of the same old_string 1 of 35 (3%; unedited 10% to 24%);
run_command 29% of 873 calls (unedited 20% to 28%); semantic search 8% (unedited 7% to 15%); failed edits 35 of 119 (29%; unedited
27% to 39%); Colab tests 8 of 58. The primary outcome did not move; the handling of failed edits did, which matches where the
edit_file -> read_file neuron fires. One seed and no control yet: EXP-084b (control neurons, seed 1) runs next on the same L4, and
seeds 2 and 3 run only if the primary outcome moves against it (D-047).

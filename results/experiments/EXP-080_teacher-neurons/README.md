# EXP-080: teacher-neurons

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T01:28:49.737406Z / 2026-11-08T01:50:05.140703Z
- **Wall time:** 21.3 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-079, EXP-076
- **Command:** `experiments/scripts/exp080_teacher_neurons.py`

## Why this experiment
Add neurons that move E2B towards what Gemma 4 31B does at the same agent state, built in closed form from EXP-079's labels.

## Hypothesis (stated before running)
Pairs (E2B action X, 31B action Y) with enough confident disagreements give read-outs that separate their states on held-out tasks far above control.

## Configuration
```json
{
  "layer": 34,
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
      "n_pos": 87,
      "n_neg": 689,
      "sep_fit": {
        "pos_mean": 0.762,
        "pos_min": 0.377,
        "neg_median": 0.0855,
        "neg_p99": 0.8586,
        "neg_p999": 0.8879,
        "neg_max": 0.8935,
        "n_pos": 87,
        "n_neg": 689
      },
      "sep_test": {
        "pos_mean": 0.757,
        "pos_min": 0.526,
        "neg_median": 0.0806,
        "neg_p99": 0.8644,
        "neg_p999": 0.8972,
        "neg_max": 0.8989,
        "n_pos": 107,
        "n_neg": 714
      },
      "control_on_test_positives": 0.5021,
      "dev_gap_median": -9.25,
      "predicted": {
        "test_positives": {
          "n": 107,
          "y_beats_x_before": 0.0,
          "after": 0.271
        },
        "test_agreeing_x_states": {
          "n": 110,
          "y_beats_x_before": 0.009,
          "after": 0.164
        }
      }
    },
    "search_similar_code_to_run_command": {
      "x": "search_similar_code",
      "y": "run_command",
      "c": 10,
      "n_pos": 58,
      "n_neg": 718,
      "sep_fit": {
        "pos_mean": 0.877,
        "pos_min": 0.35,
        "neg_median": 0.0134,
        "neg_p99": 0.951,
        "neg_p999": 1.0002,
        "neg_max": 1.0276,
        "n_pos": 58,
        "n_neg": 718
      },
      "sep_test": {
        "pos_mean": 0.886,
        "pos_min": 0.389,
        "neg_median": 0.0175,
        "neg_p99": 1.0109,
        "neg_p999": 1.0458,
        "neg_max": 1.0645,
        "n_pos": 65,
        "n_neg": 756
      },
      "control_on_test_positives": 0.4388,
      "dev_gap_median": -7.062,
      "predicted": {
        "test_positives": {
          "n": 65,
          "y_beats_x_before": 0.015,
          "after": 0.815
        },
        "test_agreeing_x_states": {
          "n": 28,
          "y_beats_x_before": 0.0,
          "after": 0.393
        }
      }
    },
    "edit_file_to_read_file": {
      "x": "edit_file",
      "y": "read_file",
      "c": 10,
      "n_pos": 45,
      "n_neg": 731,
      "sep_fit": {
        "pos_mean": 0.853,
        "pos_min": 0.634,
        "neg_median": 0.0251,
        "neg_p99": 0.8935,
        "neg_p999": 0.9178,
        "neg_max": 0.9294,
        "n_pos": 45,
        "n_neg": 731
      },
      "sep_test": {
        "pos_mean": 0.821,
        "pos_min": 0.419,
        "neg_median": 0.0212,
        "neg_p99": 0.8914,
        "neg_p999": 0.9354,
        "neg_max": 0.9414,
        "n_pos": 38,
        "n_neg": 783
      },
      "control_on_test_positives": 0.506,
      "dev_gap_median": -7.875,
      "predicted": {
        "test_positives": {
          "n": 38,
          "y_beats_x_before": 0.0,
          "after": 0.658
        },
        "test_agreeing_x_states": {
          "n": 35,
          "y_beats_x_before": 0.0,
          "after": 0.314
        }
      }
    },
    "edit_file_to_run_command": {
      "x": "edit_file",
      "y": "run_command",
      "c": 10,
      "n_pos": 30,
      "n_neg": 746,
      "sep_fit": {
        "pos_mean": 0.845,
        "pos_min": 0.657,
        "neg_median": 0.0177,
        "neg_p99": 0.9045,
        "neg_p999": 0.9444,
        "neg_max": 0.9483,
        "n_pos": 30,
        "n_neg": 746
      },
      "sep_test": {
        "pos_mean": 0.781,
        "pos_min": 0.231,
        "neg_median": 0.0097,
        "neg_p99": 0.875,
        "neg_p999": 0.9185,
        "neg_max": 0.9324,
        "n_pos": 37,
        "n_neg": 784
      },
      "control_on_test_positives": 0.4578,
      "dev_gap_median": -12.406,
      "predicted": {
        "test_positives": {
          "n": 37,
          "y_beats_x_before": 0.0,
          "after": 0.054
        },
        "test_agreeing_x_states": {
          "n": 35,
          "y_beats_x_before": 0.0,
          "after": 0.029
        }
      }
    },
    "reread_full": {
      "x": "edit_file",
      "y": "read_file",
      "c": 10,
      "n_pos": 74,
      "n_neg": 702,
      "sep_fit": {
        "pos_mean": 0.695,
        "pos_min": 0.272,
        "neg_median": 0.2847,
        "neg_p99": 0.7166,
        "neg_p999": 0.8224,
        "neg_max": 0.8397,
        "n_pos": 74,
        "n_neg": 702
      },
      "sep_test": {
        "pos_mean": 0.663,
        "pos_min": 0.041,
        "neg_median": 0.2655,
        "neg_p99": 0.7798,
        "neg_p999": 0.8428,
        "neg_max": 0.8626,
        "n_pos": 89,
        "n_neg": 732
      },
      "control_on_test_positives": 0.4454,
      "dev_gap_median": 1.906,
      "predicted": {
        "test_positives": {
          "n": 89,
          "y_beats_x_before": 0.562,
          "after": 0.775
        },
        "test_agreeing_x_states": {
          "n": 22,
          "y_beats_x_before": 0.0,
          "after": 0.182
        }
      }
    }
  },
  "reread_at_full_context": {
    "dev": {
      "pos_mean": 0.609,
      "pos_min": -0.079,
      "neg_median": 0.2995,
      "neg_p99": 0.8302,
      "neg_p999": 0.8567,
      "neg_max": 0.8654,
      "n_pos": 74,
      "n_neg": 702
    },
    "test": {
      "pos_mean": 0.601,
      "pos_min": -0.164,
      "neg_median": 0.2619,
      "neg_p99": 0.8478,
      "neg_p999": 0.9358,
      "neg_max": 0.9521,
      "n_pos": 89,
      "n_neg": 732
    }
  },
  "headline": "5 teacher neurons built; read_file_to_run_command (c=10): test positives 0.757 vs negatives p99 0.8644, control 0.5021, predicted {'test_positives': {'n': 107, 'y_beats_x_before': 0.0, 'after': 0.271}, 'test_agreeing_x_states': {'n': 110, 'y_beats_x_before': 0.009, 'after': 0.164}}; search_similar_code_to_run_command (c=10): test positives 0.886 vs negatives p99 1.0109, control 0.4388, predicted {'test_positives': {'n': 65, 'y_beats_x_before': 0.015, 'after': 0.815}, 'test_agreeing_x_states': {'n': 28, 'y_beats_x_before': 0.0, 'after': 0.393}}; edit_file_to_read_file (c=10): test positives 0.821 vs negatives p99 0.8914, control 0.506, predicted {'test_positives': {'n': 38, 'y_beats_x_before': 0.0, 'after': 0.658}, 'test_agreeing_x_states': {'n': 35, 'y_beats_x_before': 0.0, 'after': 0.314}}; edit_file_to_run_command (c=10): test positives 0.781 vs negatives p99 0.875, control 0.4578, predicted {'test_positives': {'n': 37, 'y_beats_x_before': 0.0, 'after': 0.054}, 'test_agreeing_x_states': {'n': 35, 'y_beats_x_before': 0.0, 'after': 0.029}}; reread_full (c=10): test positives 0.663 vs negatives p99 0.7798, control 0.4454, predicted {'test_positives': {'n': 89, 'y_beats_x_before': 0.562, 'after': 0.775}, 'test_agreeing_x_states': {'n': 22, 'y_beats_x_before': 0.0, 'after': 0.182}}",
  "system_monitor": {
    "samples": 22
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

## Reading (2026-11-08 01:55 UTC)

Colab G4, 21 minutes. 1,597 tool-call states at full context (E2B bf16 reference, layer 34). Neurons built where a pair had at
least 25 dev positives. Test-split numbers (tasks the fit never saw); "flips" are the first-order prediction that Y's first token
beats X's at the tool-name position with the write on; "harm" is the same at test states where E2B took X and 31B agreed.

| neuron | c | dev positives | test read-out: positives mean / negatives median / p99 | control on positives | dev logit gap Y-X, median | flips before -> after | harm |
|---|---|---|---|---|---|---|---|
| search_similar_code -> run_command | 10 | 58 | 0.886 / 0.018 / 1.011 | 0.439 | -7.06 | 1.5% -> 81.5% (n 65) | 39.3% (n 28) |
| edit_file -> read_file | 10 | 45 | 0.821 / 0.021 / 0.891 | 0.506 | -7.88 | 0% -> 65.8% (n 38) | 31.4% (n 35) |
| re-read after a failed edit (EXP-076 refitted) | 10 | 74 | 0.663 / 0.266 / 0.780 | 0.445 | +1.91 | 56.2% -> 77.5% (n 89) | 18.2% (n 22) |
| read_file -> run_command | 10 | 87 | 0.757 / 0.081 / 0.864 | 0.502 | -9.25 | 0% -> 27.1% (n 107) | 16.4% (n 110) |
| edit_file -> run_command | 10 | 30 | 0.781 / 0.010 / 0.875 | 0.458 | -12.41 | 0% -> 5.4% (n 37) | 2.9% (n 35) |

The read-outs are selective on held-out tasks (positives 0.66 to 0.89 against negative medians 0.01 to 0.27). The limit is the write
position: every neuron hit the cap c = 10 without reaching 90% flips, because at the tool-name token E2B's own thinking has usually
already chosen the action (the teacher's action trails by 7 to 12 logits). Two neurons are close to inert at this position
(read_file -> run_command 27%, edit_file -> run_command 5%). EXP-076's original read-out keeps its strength at full context (test
positives 0.601, negatives median 0.262), as EXP-081 found. A flip after the thinking can also leave the call's arguments written for
the old tool, so real-task runs must also count malformed calls.

Real-task test, seed 1 first (official E2B W4A16, Colab G4, scored on Kaggle; unedited G4 baselines EXP-044d, 044f, 044c at seeds 1
to 3): teacher_all (the four teacher neurons), control_all, neuron_search_similar_code_to_run_command, neuron_edit_file_to_read_file,
neuron_reread_full and control_reread_full. Mechanism outcomes: tool shares (run_command 9% to 13% and semantic search 15% to 23% of
calls in the four unedited G4 runs), read before edit (50% to 64% of edits), next call after a failed edit (25% to 46% read_file in
eight unedited runs). Seeds 2 and 3 only for arms whose seed-1 mechanism outcome moves.

## First real-task run (2026-11-08 02:12 UTC)

EXP-080a, teacher_all (the four teacher neurons, c = 10), seed 1, official E2B W4A16 on a Colab G4; official score pending on
Kaggle. Mechanism outcomes against the four unedited G4 runs (EXP-044c, 044d, 044e, 044f):
- run_command 17% of 629 tool calls (unedited 9% to 13%); search_similar_code 13% (unedited 15% to 23%); read_file 45% (33% to 35%);
  edit_file 14% (19% to 24%).
- edits preceded by reading the same file: 65 of 88 (74%; unedited 50% to 64%).
- tasks running pytest after their last applied edit: 34 of 43 (79%; unedited 56% to 72%).
- loops unchanged: 22 of 58 tasks, 35 loops (unedited 18 to 28 tasks); median output 5,854 tokens per task.
- failed edits 46 of 88 (52%; unedited 46% to 66%).
Every pre-registered mechanism outcome moved beyond the unedited range, in 31B's direction, with no loop damage. The control with the
same writes and random read-outs (EXP-080b) decides whether the selectivity of the read-outs matters.

EXP-080b, control_all (same writes, read-outs fitted to random states), seed 1: inside the unedited range on every outcome.
run_command 14% of 490 calls (teacher 17%, unedited 9% to 13%); read_file 35% (teacher 45%); edit_file 20% (teacher 14%); read
before edit 58 of 96 (60%; teacher 74%); pytest after the last edit 32 of 45 (71%; teacher 79%); loops 17 of 58 tasks. At seed 1 the
behavioural change is the selective read-outs' doing, not a broad bias of the writes. Seeds 2 and 3 of teacher_all and control_all
were queued (EXP-080g to j), as D-047 allows when the seed-1 mechanism outcome moves.

EXP-080c (re-read neuron refitted at full context, c = 10, seed 1) against EXP-080d (its control): after a failed edit, next call
read_file 21 of 47 (45%) against 23 of 62 (37%), unedited 25% to 46%; resends of the same old_string 4 of 47 (9%) against 13 of 62
(21%), unedited 14% to 31%; failed edits 52 of 103 (50%) against 66 of 108 (61%); loops 19 of 58 tasks. The pre-registered outcome
(re-reading after a failed edit) stays inside the unedited range, so no further seeds under D-047. The one sign beyond the unedited
range is fewer blind retries (9%), a single-seed observation on 47 failed edits, not a claim.

EXP-080e (the single edit_file -> read_file neuron, c = 10, seed 1): read before edit 56 of 89 (63%; unedited 50% to 64%), edit_file
19% and read_file 37% of 472 calls (unedited 19% to 24% and 33% to 35%), loops normal. Alone it stays inside the unedited range;
with the other three teacher neurons (EXP-080a) the same outcome reached 74%. No further seeds for this single neuron (D-047).

EXP-080g (teacher_all, seed 2): read_file 57% of 715 calls (seed 1: 45%; unedited seed 2, EXP-044f: 34%), edit_file 12% (14%; 19%),
search_similar_code 12% (13%; 23%), run_command 8% (17%; 10%); read before edit 64 of 87 (74%; 74%; 52%); after a failed edit the next
call is a read in 26 of 53 (49%); blind retries 5 of 53 (9%; unedited seed 2: 31%); tasks with no patch 11 (seed 1: 3; unedited
seed 2: 2); model calls 859 (unedited 702); loops 22 of 58 tasks. Consistent over two seeds: more reading before editing, fewer
edits, less semantic search, fewer blind retries. Not consistent: the shift to run_command (seed 1 only). At seed 2 the extra
reading left 11 tasks with no patch, a likely cost to the solved count.

EXP-080f (the single search_similar_code -> run_command neuron, c = 10, seed 1; 57 of 58 tasks finished, the last stopped at 35
minutes): search_similar_code 16% and run_command 10% of 517 calls (unedited 15% to 23% and 9% to 13%), read before edit 60 of 120
(50%), failed edits 83 of 120 (69%; unedited 46% to 66%). No shift, although the build predicted 82% flips at its positives. Both
single teacher neurons (080e, 080f) stay inside the unedited range; only the four together (080a, 080g) move behaviour, so the
first-order prediction from the logit gap overstates a single neuron's effect in the served W4A16 model. No further seeds for the
single neurons (D-047).

EXP-080h (control_all, seed 2): inside the unedited range again: read_file 36%, edit_file 21%, search_similar_code 15%, run_command
12% of 576 calls; read before edit 70 of 120 (58%); blind retries 9 of 60 (15%); tasks with no patch: see the record. Over two seeds,
each against its own control, the four teacher neurons raise reading before editing (74% and 74% against 60% and 58%), cut edit
calls (14% and 12% against 20% and 21%) and cut semantic search (13% and 12% against 16% and 15%).

EXP-080i (teacher_all, seed 3) against EXP-080j (control_all, seed 3): read before edit 73 of 91 (80%) against 63 of 104 (61%);
failed edits 44 of 91 (48%) against 55 of 104 (53%); blind retries 4 of 40 (10%) against 16 of 53 (30%); edit_file 13% against 21%,
search_similar_code 11% against 16% of calls; context overflows 21 against 15.

Pooled over seeds 1 to 3, each arm against its own control: read before edit 202 of 266 edits (76%) against 191 of 320 (60%);
blind retries 20 of 138 failed edits (14%) against 40 of 165 (24%); failed edits 144 of 266 (54%) against 175 of 320 (55%).
Task-level test of the primary mechanism outcome: on the 54 tasks with edits in both arms, the per-task share of edits preceded by a
read of the same file rose by 0.19 on average with the teacher neurons; higher in 35 tasks, lower in 12; sign-flip p = 0.0001
(100,000 random sign patterns). The behavioural effect is established. Official solved counts for seeds 2 and 3 are pending.

Official solved counts (Kaggle, 04:15 UTC). teacher_all 4, 4, 5 (13 over seeds 1 to 3); control_all 1, 4, 4 (9); unedited G4 runs at
the same seeds 6, 4, 4 (14). Pooled task-level sign-flip tests: teacher against control +4 solves, 5 tasks differ, p = 0.31;
teacher against unedited -1, 6 tasks differ, p = 1.0. The pre-registered score outcome shows no gain. The behavioural change
(D-049) did not convert into tasks solved on E2B.

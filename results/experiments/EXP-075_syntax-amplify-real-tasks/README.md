# EXP-075: amplifying the tool-call syntax neurons, tested on real tasks

- **Status:** pre-registered 2026-11-07 21:10 UTC, before any run. Runs on Google Colab L4s through tools/colab_batch.py.
- **Category:** improvement. **Depends on:** EXP-062 (neuron atlas), EXP-063 (causal ablation).

## Why
In EXP-063 (32 saved edit moments, Q4_0, seed 0), switching off the 2,480 neurons of atlas groups 3, 12, 1 and 4 lowered the share of
edit calls that applied from 20 of 27 to 11 of 30, and "old_string not found" rose from 5 to 14. Switching off 2,480 matched random
neurons gave 16 of 26. Failed edits are the largest mechanical failure on real tasks: 51 of 110 edit calls failed in EXP-044a.
This experiment asks whether the opposite move, amplifying the group, raises exact copying and tasks solved.

## Arms (fixed now)
- `amp_syntax_a150`: the group's down_proj columns times 1.5.
- `amp_syntax_a125`: the same, times 1.25.
- `amp_rand_a150`: EXP-063's matched random neurons (seed 63, same count per layer) times 1.5. This is the control.
- No edit: EXP-044a (seed 1, Kaggle), EXP-044b and EXP-044c (seeds 2 and 3, Colab).

Each arm runs at seeds 1, 2 and 3 on the 58 small-fix tasks, official settings. The adapters are exact LoRAs: every module is padded
to rank 452 with lora_alpha 452, so the scale is 1 in any loader. Each stored change was checked against the intended one
(relative error under 0.002 at float16, see `edit_*.json`).

## Measures (fixed now)
- Primary: tasks solved, summed over the 3 seeds, for each arm against no edit and against the control.
- Secondary: share of edit calls that applied (from traces), "not found" failures, context errors, loops, tool calls per task.
- A gain counts only if it beats both no edit and the control, and holds across seeds by task (sign-flip test on per-task solved
  counts). Seeds are few, so a difference of one or two tasks is not read as an effect.

## Results
Pending.

## Serving speed, read from the traces (2026-11-08 01:05 UTC)

EXP-075a (amp_syntax_a150, seed 1, Colab L4) ran twice as slow as unedited E2B on the same GPU type, and the cause is the serving,
not the model's behaviour. Per model call: median output 276 tokens against 288 (EXP-044b, unedited) and 343 (EXP-058b, a rank-1
edit); median throughput 25.2 tokens per second of wall time against 51.6 and 48.4. Loops were unchanged (22 of 58 tasks against 21
in EXP-044b). The adapter is rank 452, which adk_submission serves as vLLM rank 512, and the LoRA kernels at that rank halve the
speed. In 075a this caused one request time-out and one task cut by the run's time limit. amp_rand_a150 has the same rank, so the
syntax-against-random comparison is fair; the comparison with unedited runs carries a small time confound. Cut before those runs
started: the a125 arm and seed 3 (075c, f, g, h, i). Kept: syntax and random at seeds 1 and 2 (075a, b, d, e).

Stopped 2026-11-08 02:08 UTC under D-047's rule (seeds 2 and 3 only where the seed-1 mechanism outcome moves): at seed 1 the syntax
arm failed 61 of 117 edits (52%) and the random arm 52 of 104 (50%), unedited EXP-044b 50 of 103 (49%). EXP-075d was stopped two
minutes after launch and EXP-075e removed; worker 2's L4 runs unedited E4B seeds 4 and 5 instead (EXP-073c, 073d).

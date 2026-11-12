# Neuron scan: find the neurons that track an agent's success, in any model

The tool behind EXP-154 (31B) and EXP-158 (12B). It finds which MLP neurons of a model separate the agent's runs that will solve a
task from runs that will not, at a chosen decision moment, and tests the finding on tasks it was not chosen on.

## What it needs

1. **The agent's own runs, with outcomes.** Official harness runs with request logging on (`log_requests` in the run config), so
   vLLM writes every request's exact prompt and output token ids; and a verifier score per run (tools/kaggle_verify.py). Several runs
   per task, with both outcomes on some tasks: the scan compares a passing and a failing run of the same task, so task difficulty
   cannot explain a neuron.
2. **The model in full precision** (Hugging Face weights, any decoder whose layers have an MLP with a `down_proj`; Gemma, Llama,
   Qwen and Mistral families all do).

## The four steps

1. **Bench** (`tools/analysis/build_success_bench.py`, CPU, minutes): matches each logged request to its agent step and keeps,
   per run and task, the request at the decision moment: `--point first_edit` (the first edit to a file the reference fix touches)
   or `--point submit` (the last submit). Label: did that run's patch pass.
2. **Scan** (`experiments/scripts/exp154_31b_success_neurons.py` with `SCAN_MODEL`, `SCAN_BENCH`, `SCAN_ID`; one GPU, about 15
   minutes for 123 requests on 31B): replays every request through the model (teacher forcing, no generation) and records every
   neuron's activation entering `down_proj` at two positions, the last prompt token and the mean over the output. For 31B that is
   60 layers x 21,504 neurons x 2 = 2,580,480 features.
3. **Test** (same script): each feature's within-task pairwise AUC; the 64 furthest from 0.5 on half of the tasks are combined and
   scored on the other half (two folds); the null repeats the whole selection on labels shuffled within tasks (200 times).
4. **Edit** (`tools/analysis/build_neuron_scale_adapter.py`): the chosen neurons' outputs scaled by `1 + s * alpha` (s = +1 for a
   neuron higher in passing runs, -1 for one higher in failing runs), as exact LoRAs on `down_proj`: one-hot rows in A, the neuron's
   own output column times `s * alpha` in B. The edit changes those neurons and nothing else, and ships as an adapter on the
   official base.

## Results so far

- **31B (EXP-154):** 64 neurons in 38 of 60 layers, held-out AUC 0.65 against a selection-matched null median of 0.48 (p = 0.035).
  Doubling or zeroing them (EXP-155a) solved 9 of 14 tasks against 8 to 12 unedited: they mark success but the blunt edit did not
  cause it. EXP-156/157 test six variants.
- **12B (EXP-158):** held-out AUC 0.71 over 24 pairs (p = 0.13): the same direction, too few pairs.
- **The six edits of 31B's 64 (EXP-155a to 157):** 57 of 88 task-runs against 65 of 90 unedited. Markers, not levers.

## Step 5: causal neurons (EXP-160, `experiments/scripts/exp160_31b_causal_neurons.py`)

Correlation finds markers. To find causes, pair a failing run's prefix with its own edit call and with a passing run's edit call
on the same task, take the preference (mean log-probability per token of the passing call minus the failing one), and rank every
neuron by gradient times activation of that preference. Validate by actually scaling the chosen neurons on held-out pairs against
the same number of random neurons. On 31B: 7 of 7 held-out pairs raised (random 4 of 7; p = 0.016), by +0.12 nats per token, but
the gap was 0.46, so in the agent (EXP-161a) they solved 9 of 15 (unedited 9 to 13). Always report the effect against the gap it
must close. With all pairs (EXP-162b, `experiments/scripts/exp162_31b_causal_dose.py`, prompts cut to their first 1,024 and last 4,096
tokens): 20 of 25 held-out pairs raised (random 8; p = 0.004 by pair, 0.15 by task). The dose test there closes 44% of the gap with 64
neurons x2, 57% with 256 x2 (damage 0.10 nats per token on the model's own text) and 71% with 256 x4 (damage 0.79): the effect
saturates. Memory: on a 96 GB G4 the pass grows about 4.5 MB per prompt token over the 61 GB bf16 31B, so prompts over about 8,000
tokens need cutting (or a smaller model, or offloading).

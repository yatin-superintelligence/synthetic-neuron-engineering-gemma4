# Synthetic Neuron Engineering of a Gemma 4 Coding Agent and the Decision Gap

Code, experiment records and artifacts for the paper of the same name by Yatin Taneja (November 2026), submitted to the
Google Gemma 4 Developer Agent Paper Track on Kaggle. The paper is in [`paper/`](paper/).

## Summary

Synthetic neuron engineering, also called weight-space programming, changes a network's behaviour by computing new weights
directly instead of training them. We apply it to Gemma 4 working as a software-engineering agent that must fix real bugs so
that hidden tests pass.

- **Behaviour can be steered precisely, without training.** Against matched controls, computed weight changes stopped the
  agent's reasoning from looping, made it read a file before changing it, repaired a formatting fault in the largest model's
  tool calls, and, by transplanting neurons from a larger model, made the smallest model better at predicting the correct fix.
- **No intervention raised the number of bugs fixed,** and two lowered it.
- **Why:** in the largest model, the neurons that predict a successful run are not the ones that produce it. The neurons that do
  push the model toward a correct fix can be found reliably, but they close only about half of the *decision gap*, the gap
  between the wrong fix the model prefers and the correct one, before they damage its output.
- **A measure for other work:** the decision gap is measured in minutes from recorded runs, before any agent run, and can screen
  any post-training method.

Everything in the paper can be traced from this repository: each claim to the experiments behind it, each experiment to its
script, its configuration, its results and, for agent runs, its official verification.

## Layout

| Path | Contents |
|---|---|
| `paper/` | The paper (PDF), with the full reference list and Appendix A. |
| `code/sne/` | The library (28 modules): engineered neurons and their write vectors (`conditional.py`), loop directions (`loops.py`), the transplant (`transplant.py`), adapters (`lora.py`), read-outs, statistics. |
| `code/experiments/` | One script per experiment (113 scripts), named `expNNN_...`. |
| `code/analysis/` | The scripts that compute the paper's numbers and figures (38 scripts). |
| `results/experiments/` | The record of 213 experiments, one folder each: `README.md` (aim, hypothesis stated before the run, outcome), `config.json`, `results.json`, and where they exist `kaggle_verify.json` (the official verification), `summary.json`, `metrics.json` and `run.log`. |
| `results/experiments.csv` | Index of those experiments: id, folder, category, start time, duration, status. |
| `results/agent_runs.csv` | The 129 agent runs scored with the competition's official verification (5102 task attempts): model, tasks, tasks solved, and which. Runs made in rounds have no folder of their own. |
| `results/official_verification.csv` | The same verification, one row per attempt: run, task, solved by the official verification, solved by the run's own tests. |
| `results/neurons/` | The neuron lists for the largest model (see below). |
| `results/dose_test_summary.json` | The dose test of the paper's Table 2 and Figure 3. |
| `results/figures/` | The paper's three figures. |
| `adapters/` | Four adapters in PEFT LoRA format (see below). |
| `docs/NEURON_SCAN.md` | Method note for the predictive scan and the causal search. |

## Where the Evidence for Each Claim Is

| Paper | Claim | Experiments | Start here |
|---|---|---|---|
| Section 5, Figure 1 | Where the small models' attempts end | all E2B and E4B agent runs | `code/analysis/localization.py` |
| Section 6, Table 1 | Loop direction removed (E2B); rebuilt for 12B | EXP-053, EXP-058; EXP-133 | `code/sne/loops.py` |
| Section 6, Table 1 | Neurons imitating 31B's choices | EXP-080 | `code/sne/conditional.py` |
| Section 6, Table 1 | Transplant from E4B into E2B | EXP-102, EXP-106, EXP-108 | `code/experiments/exp102_neuron_transplant.py` |
| Section 6, Figure 2 | Quoting sentence forced (Test 1); weight change in its place (Test 2) | EXP-116; EXP-120, EXP-121 | `code/experiments/` (script named after the experiment) |
| Section 6 | Instruction written into the weights | EXP-104 | `code/experiments/exp104_instruction_to_neurons.py` |
| Section 7 | Tool-call repair with a gated neuron (31B) | EXP-140, EXP-144; agent rounds EXP-146R0 | `code/experiments/` (script named after the experiment) |
| Section 7 | Predictive neurons: scan and agent runs | EXP-154; agent runs EXP-155A, EXP-156R0, EXP-156R1, EXP-156R2, EXP-157R0, EXP-157R1 (57 of 88 attempts solved) | `code/experiments/exp154_31b_success_neurons.py, docs/NEURON_SCAN.md, results/neurons/predictive_64.json, results/agent_runs.csv` |
| Section 7, Table 2, Figure 3 | Causal neurons: two searches, dose test, agent runs | EXP-160, EXP-161, EXP-162, EXP-162b, EXP-163 | `code/experiments/exp160_31b_causal_neurons.py, exp162_31b_causal_dose.py, results/neurons/, results/dose_test_summary.json, adapters/` |
| Section 3, Table A4 | Settings and prompts tested on the baseline model | EXP-115, EXP-122, EXP-138, EXP-139, EXP-141, EXP-143, EXP-149, EXP-150, EXP-153 | `results/agent_runs.csv` |
| Table A4 | Trained comparison (LoRA fine-tuning) | EXP-088, EXP-110 | `code/experiments/exp088_sft_lora.py` |

Experiment folders in `results/experiments/` begin with the experiment's id (for example `EXP-162b_...`); a letter after the
number marks one run of that experiment.

## The Neuron Lists

- `predictive_64.json`: the 64 predictive neurons of Section 7. `layer`, `neuron` (index among the layer's 21,504 feed-forward
  neurons), `position` (where the activation is read: `last` is the last prompt token, `outmean` the mean over the output
  tokens) and `auc` (within-task AUC in the scan that selected it; below 0.5 means higher in failing runs).
- `causal_64_first_search.json`: the 64 causal neurons of the first search. `action` is `double` or `silence`; `attribution` is
  the score defined in Section 4. This set was run in the agent (9 of 15 tasks solved).
- `causal_256_second_search.json`: the 256 causal neurons of the second search, with `in_64_set` true for the 32 highest and
  the 32 lowest attributions. The full set, doubled and silenced, was run in the agent (10 of 15).
- `dose_test_summary.json`: 25 held-out cases from 12 tasks. `chosen_32` is 64 neurons (32 doubled, 32 silenced), `chosen_128`
  is 256; `x2` doubles and silences, `x4` multiplies by four and silences. For each: cases in which the preference for the
  correct fix rose, its mean change and the damage (nats per token), and the cases in which the correct fix is the preferred
  one (`base_flips`: with no intervention). The mean decision gap is 0.4565 nats per token.

Layers are numbered from 0; 31B has 60 layers.

## The Adapters

| Folder | Model | What it is |
|---|---|---|
| `adapters/causal_64_first_search/` | Gemma 4 31B | The first search's 64 causal neurons, 32 doubled and 32 silenced. Run in the agent: 9 of 15 tasks solved. |
| `adapters/causal_256_second_search/` | Gemma 4 31B | The second search's 256 causal neurons, 128 doubled and 128 silenced. Run in the agent: 10 of 15. |
| `adapters/transplant_e2b/` | Gemma 4 E2B | The transplant of 896 neurons from E4B (7 layers, 128 neurons each), as built by `code/experiments/exp102_neuron_transplant.py`. |
| `adapters/transplant_e2b_random_control/` | Gemma 4 E2B | Its control: random E4B neurons transplanted the same way. |

Scaling neuron i of layer l by (1 + s) changes only column i of that layer's `down_proj` weight:
W_down[:, i] <- (1 + s) * W_down[:, i]. As a LoRA, lora_A holds the one-hot row e_i and lora_B the column s * W_down[:, i];
s = 1 doubles the neuron and s = -1 silences it. Nothing is trained. The two causal adapters were checked against their neuron
lists before release: every used row of lora_A is one-hot at a listed neuron, and the matching column of lora_B equals plus or
minus the model's own column. In the agent runs the adapters were served with vLLM 0.19.1 on the `-it-qat-w4a16-ct` checkpoints.

## How Results Were Scored

An agent run is scored by the competition's own verification code on Kaggle's official image. `kaggle_verify.json` holds that
count (`resolved` of `items`) and the solved task ids; `colab_resolved` is the count from the run's own tests, which is not the
official one and is kept only for comparison. Counts in the paper are the official ones.

## What Is Not Here, and Why

- The competition's data: task statements, repositories, tests and the official notebook and prompt. Recorded agent runs are
  left out for the same reason: they contain task content. The code therefore cannot be run from this repository alone; it is
  shared so that every step can be read and checked.
- Adapters for the other interventions and intermediate activations (several gigabytes). Available on request.
- The project's internal decision log. Experiment records refer to its entries as `D-` numbers and to `RESEARCH_LOG.md`; the
  aim, the hypothesis stated before the run and the outcome are in each experiment's own `README.md`.

A few records and scripts were left out because they quote the official prompt or hold machine-specific details. In the
remaining files the local project path is written `<repo>`.

## Citation

Yatin Taneja. *Synthetic Neuron Engineering of a Gemma 4 Coding Agent and the Decision Gap.* Google Gemma 4 Developer Agent
Paper Track, Kaggle, November 2026.

The adapters contain columns of Gemma 4's own weights and are shared under that model's terms of use.

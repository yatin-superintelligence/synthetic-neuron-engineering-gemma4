# EXP-037: official E2B baseline on Kaggle, default budget

- **Status:** OK. This is the E2B baseline on real tasks. EXP-036 is the same run with a 5-minute limit.
- **Stack:** private Kaggle notebook sne-exp037-e2b-official-default-budget (starter notebook, minimally changed), L4 GPUs,
  vLLM, gemma-4-e2b-it-qat-w4a16-ct, swegemma 0.2.11, subprocess sandbox, sample submission agent, no adapters, concurrency 12.
- **Budgets:** harness defaults: 60 minutes, 100 tool calls, 300 s per command, thinking budget 4096, output limit 16384.
- **Tasks:** the same 58 public tasks with a single-file reference patch of at most 10 changed lines.
- **Wall time:** 1419 s of notebook time. Quota went from 4.67 to 5.46 h, so a full run costs about 0.8 h. Mean task time was 3.0 minutes.

## Results
Resolved 2/58 (3.4%): fastapi_14786, requests_6644. Both also resolved in EXP-036, which resolved 4/58.
Outcomes: patch failed tests 32, context window error 15, no patch 9, resolved 2. Mean 8.3 tool calls, 10.8 LLM calls.
Thinking loops: 26 responses in 17 tasks, none resolved, 31% of all completion tokens.
Edits: 74 of 132 edit_file calls failed; 17 tasks never applied an edit and none of those resolved.
Numbers are in `summary.json`, recomputed by `tools/trace_summary.py`.

## Reading
- **Correction to EXP-036.** Its README says the 5-minute limit made 6.9% an understatement. This run does not support that:
  with the limit removed, tasks stopped after 3 minutes on average and 2 resolved. The difference of 2 tasks is within
  sampling noise, which is not yet measured for this stack. Budget was not the constraint for E2B on these tasks.
- **Budget is unused.** The agent spends 8 of 100 allowed tool calls. It stops early, so it does not use time to verify.
- **Loops cause context errors.** 10 of the 15 context errors follow a loop. In several, the response passes the 4096-token
  thinking budget and runs on in the visible channel to the 16384 output limit, which fills the 32768 window. This is a
  stronger failure than H-3 first described, and the output limit is a free setting to test.
- **Argument order.** Across EXP-036 and EXP-037 together, edits with old_string first applied 9 of 10 times, new_string
  first 91 of 207 (Fisher p = 0.006). This is observational. EXP-041 tests it by intervention.
- Resolved rate is too low on 58 tasks to compare small effects in a single run, so any later comparison uses repeated runs.

Competition data and traces are not stored in this repo; the bundle is kept in artifacts_local.

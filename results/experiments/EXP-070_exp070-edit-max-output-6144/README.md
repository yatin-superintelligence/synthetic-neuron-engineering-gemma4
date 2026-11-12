# EXP-070: official E2B, loop edit with max_output_tokens 6144, seed 1

- **Status:** OK, recorded automatically by tools/kaggle_batch.py. Read critically in RESEARCH_LOG.md.
- **Model:** gemma-4-e2b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** the 58 public small-fix tasks. Unfinished tasks count as unresolved.
- **Wall time:** about 1948 s.

## Results
4/58 resolved of 58 finished; {'patch failed tests': 44, 'no patch': 10, 'resolved': 4}; 81/174 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

## Reading

Compared with EXP-058a, the same edit and seed with the official output limit of 16,384 tokens.

- Context errors fell from 13 of 58 tasks to 0 of 58. The error text gives the cause: 16,384 reserved output tokens plus a prompt of at least 16,385 exceed the 32,768 limit. At 6144, prompts can reach 26,624 tokens.
- Solved tasks went from 5 of 58 to 4 of 58: 3 solved by both, 2 lost, 1 gained. One seed, so no effect is shown.
- Of the 13 tasks that had context errors, 12 now submit a patch that fails its tests, 1 submits nothing, and 0 are solved.
- All 10 no-patch tasks called submit_patch, were told patch_size 0 and files_changed 0, and then stopped.
- Of the 93 failed edit_file calls, a stricter matcher could place 31 at one clear location. 42 were weak or ambiguous, 19 shared no exact line with the file, and 1 named a missing file. Placing an edit does not show the fix was right.

Scripts are in `analysis/`.

# EXP-043a: official E2B, include_thoughts false, seed 1

- **Status:** OK, recorded automatically by tools/kaggle_batch.py. Read critically in RESEARCH_LOG.md.
- **Model:** gemma-4-e2b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** the 58 public small-fix tasks. Unfinished tasks count as unresolved.
- **Wall time:** about 2658 s.

## Results
4/58 resolved of 58 finished; {'patch failed tests': 31, 'error: Sandbox execution error: litellm.Context': 14, 'no patch': 8, 'resolved': 4, 'error: Agent completed execution without callin': 1}; 62/336 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

# EXP-049: official E2B, both prompt changes

- **Status:** OK, recorded automatically by tools/kaggle_batch.py. Read critically in RESEARCH_LOG.md.
- **Model:** gemma-4-e2b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** the 58 public small-fix tasks. Unfinished tasks count as unresolved.
- **Wall time:** about 1562 s.

## Results
4/58 resolved of 58 finished; {'patch failed tests': 31, 'error: Sandbox execution error: litellm.Context': 21, 'resolved': 4, 'no patch': 2}; 49/106 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

# EXP-044f: official E2B, no edit, seed 2 (Colab G4, 16 tasks at once), the same-hardware baseline for seed 2

- **Status:** OK, recorded automatically by tools/colab_batch.py. Read critically in RESEARCH_LOG.md.
- **Where:** Google Colab, G4, official stack (the Kaggle notebook's code, with input and working folders under /content).
- **Model:** gemma-4-e2b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** small. Unfinished tasks count as unresolved.
- **Wall time:** about 528 s.

## Results
Colab tests (not official) 5/58 resolved of 58 finished; {'patch failed tests': 32, 'error: Sandbox execution error: litellm.Context': 18, 'resolved': 5, 'no patch': 2, 'error: Agent completed execution without callin': 1}; 46/102 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

**The resolved count above is not the official score.** Colab's test environment does not match the Kaggle image the official scorer uses (25 of 58 reference fixes pass there). Agent behaviour (tool calls, edits, loops, patches) is valid; whether a patch solves its task is decided by tools/kaggle_verify.py on a Kaggle CPU notebook and recorded in `kaggle_verify.json`.

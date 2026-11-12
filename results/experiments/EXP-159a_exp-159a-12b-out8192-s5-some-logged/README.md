# EXP-159a: official 12B unedited, 8,192 setting, seed 5, the 15 tasks 12B solves only sometimes, with request logging (Colab G4); bench runs for the 12B neuron scan EXP-158 (D-217)

- **Status:** OK, recorded automatically by tools/colab_batch.py. Read critically in RESEARCH_LOG.md.
- **Where:** Google Colab, G4, official stack (the Kaggle notebook's code, with input and working folders under /content).
- **Model:** gemma-4-12b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** ['fastapi_11194', 'fastapi_13537', 'fastapi_14301', 'fastapi_14463', 'fastapi_14794', 'fastapi_15589', 'fastapi_5077', 'fastapi_9425', 'fastapi_9753', 'rich_2725', 'rich_2943', 'rich_3067', 'rich_3470', 'rich_3480', 'rich_3894']. Unfinished tasks count as unresolved.
- **Wall time:** about 3742 s.

## Results
Colab tests (not official) 5/15 resolved of 15 finished; {'resolved': 5, 'patch failed tests': 4, 'error: Agent exceeded tool call budget (100 cal': 2, 'error: Sandbox execution error: litellm.Context': 2, 'error: Agent exceeded session timeout (60.0 min': 1, 'error: Agent completed execution without callin': 1}; 23/30 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

**The resolved count above is not the official score.** Colab's test environment does not match the Kaggle image the official scorer uses (25 of 58 reference fixes pass there). Agent behaviour (tool calls, edits, loops, patches) is valid; whether a patch solves its task is decided by tools/kaggle_verify.py on a Kaggle CPU notebook and recorded in `kaggle_verify.json`.

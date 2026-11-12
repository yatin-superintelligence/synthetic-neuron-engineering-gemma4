# EXP-121a: official E2B with EXP-120's inj_q_L14, the 12 ever-solved tasks, seed 2 (Colab L4)

- **Status:** OK, recorded automatically by tools/colab_batch.py. Read critically in RESEARCH_LOG.md.
- **Where:** Google Colab, L4, official stack (the Kaggle notebook's code, with input and working folders under /content).
- **Model:** gemma-4-e2b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** ['fastapi_14786', 'rich_2943', 'requests_6644', 'fastapi_14301', 'rich_3006', 'rich_3454', 'fastapi_9425', 'rich_4077', 'rich_3894', 'fastapi_9753', 'fastapi_14463', 'fastapi_14458']. Unfinished tasks count as unresolved.
- **Wall time:** about 1588 s.

## Results
Colab tests (not official) 1/12 resolved of 12 finished; {'patch failed tests': 8, 'error: Sandbox execution error: litellm.Context': 3, 'resolved': 1}; 16/24 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

**The resolved count above is not the official score.** Colab's test environment does not match the Kaggle image the official scorer uses (25 of 58 reference fixes pass there). Agent behaviour (tool calls, edits, loops, patches) is valid; whether a patch solves its task is decided by tools/kaggle_verify.py on a Kaggle CPU notebook and recorded in `kaggle_verify.json`.

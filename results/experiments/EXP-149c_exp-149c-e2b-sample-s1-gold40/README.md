# EXP-149c: official E2B unedited at the sample settings (temperature 0.2, 16,384-token answer reservation), the 40 reference-resolvable tasks, seed 1 (Colab L4, 8 at once); the baseline for EXP-149d (D-207)

- **Status:** OK, recorded automatically by tools/colab_batch.py. Read critically in RESEARCH_LOG.md.
- **Where:** Google Colab, L4, official stack (the Kaggle notebook's code, with input and working folders under /content).
- **Model:** gemma-4-e2b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** ['fastapi_11194', 'fastapi_13537', 'fastapi_13920', 'fastapi_14258', 'fastapi_14301', 'fastapi_14458', 'fastapi_14463', 'fastapi_14479', 'fastapi_14487', 'fastapi_14492', 'fastapi_14786', 'fastapi_14794', 'fastapi_15589', 'fastapi_5077', 'fastapi_5624', 'fastapi_9425', 'fastapi_9753', 'requests_6644', 'rich_2725', 'rich_2943', 'rich_3006', 'rich_3043', 'rich_3063', 'rich_3067', 'rich_3105', 'rich_3278', 'rich_3454', 'rich_3470', 'rich_3471', 'rich_3472', 'rich_3480', 'rich_3518', 'rich_3718', 'rich_3882', 'rich_3894', 'rich_3905', 'rich_3944', 'rich_4076', 'rich_4077', 'rich_4079']. Unfinished tasks count as unresolved.
- **Wall time:** about 1467 s.

## Results
Colab tests (not official) 0/40 resolved of 40 finished; {'patch failed tests': 22, 'error: Sandbox execution error: litellm.Context': 15, 'no patch': 3}; 31/72 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

**The resolved count above is not the official score.** Colab's test environment does not match the Kaggle image the official scorer uses (25 of 58 reference fixes pass there). Agent behaviour (tool calls, edits, loops, patches) is valid; whether a patch solves its task is decided by tools/kaggle_verify.py on a Kaggle CPU notebook and recorded in `kaggle_verify.json`.

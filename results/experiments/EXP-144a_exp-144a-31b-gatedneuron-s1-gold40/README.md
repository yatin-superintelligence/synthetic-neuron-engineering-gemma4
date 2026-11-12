# EXP-144a: official 31B with the gated string-end neuron (EXP-144: one replaced unit in the last MLP, rank-1 LoRAs on gate, up and down), the 8,192-token answer reservation, the 40 reference-resolvable tasks, seed 1, request logging (Colab G4, 16 at once); against EXP-139b and the zero-adapter control EXP-140e

- **Status:** OK, recorded automatically by tools/colab_batch.py. Read critically in RESEARCH_LOG.md.
- **Where:** Google Colab, G4, official stack (the Kaggle notebook's code, with input and working folders under /content).
- **Model:** gemma-4-31b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** ['fastapi_11194', 'fastapi_13537', 'fastapi_13920', 'fastapi_14258', 'fastapi_14301', 'fastapi_14458', 'fastapi_14463', 'fastapi_14479', 'fastapi_14487', 'fastapi_14492', 'fastapi_14786', 'fastapi_14794', 'fastapi_15589', 'fastapi_5077', 'fastapi_5624', 'fastapi_9425', 'fastapi_9753', 'requests_6644', 'rich_2725', 'rich_2943', 'rich_3006', 'rich_3043', 'rich_3063', 'rich_3067', 'rich_3105', 'rich_3278', 'rich_3454', 'rich_3470', 'rich_3471', 'rich_3472', 'rich_3480', 'rich_3518', 'rich_3718', 'rich_3882', 'rich_3894', 'rich_3905', 'rich_3944', 'rich_4076', 'rich_4077', 'rich_4079']. Unfinished tasks count as unresolved.
- **Wall time:** about 5119 s.

## Results
Colab tests (not official) 15/40 resolved of 40 finished; {'patch failed tests': 22, 'resolved': 15, 'error: Agent exceeded session timeout (60.0 min': 2, 'no patch': 1}; 99/298 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

**The resolved count above is not the official score.** Colab's test environment does not match the Kaggle image the official scorer uses (25 of 58 reference fixes pass there). Agent behaviour (tool calls, edits, loops, patches) is valid; whether a patch solves its task is decided by tools/kaggle_verify.py on a Kaggle CPU notebook and recorded in `kaggle_verify.json`.

# EXP-161a: official 31B with EXP-160's 64 causally chosen MLP neurons scaled (32 doubled, 32 zeroed; down_proj LoRAs), the 8,192 setting, seed 1, the 15 sometimes-solved tasks (Colab G4, 15 at once); against the six unedited 8,192 runs (9 to 13, seed 1: 11) and the six correlation-chosen edits (57 of 88) (D-222)

- **Status:** OK, recorded automatically by tools/colab_batch.py. Read critically in RESEARCH_LOG.md.
- **Where:** Google Colab, G4, official stack (the Kaggle notebook's code, with input and working folders under /content).
- **Model:** gemma-4-31b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** ['fastapi_13537', 'fastapi_14463', 'fastapi_15589', 'fastapi_5624', 'fastapi_9425', 'fastapi_9753', 'rich_2725', 'rich_3043', 'rich_3063', 'rich_3067', 'rich_3278', 'rich_3470', 'rich_3480', 'rich_4077', 'rich_4079']. Unfinished tasks count as unresolved.
- **Wall time:** about 3831 s.

## Results
Colab tests (not official) 5/15 resolved of 15 finished; {'patch failed tests': 9, 'resolved': 5, 'error: Agent exceeded session timeout (60.0 min': 1}; 36/59 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

**The resolved count above is not the official score.** Colab's test environment does not match the Kaggle image the official scorer uses (25 of 58 reference fixes pass there). Agent behaviour (tool calls, edits, loops, patches) is valid; whether a patch solves its task is decided by tools/kaggle_verify.py on a Kaggle CPU notebook and recorded in `kaggle_verify.json`.

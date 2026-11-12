# EXP-163a: official 31B with EXP-162b's 256 causally chosen MLP neurons (128 doubled, 128 zeroed; rank-16 down_proj LoRAs in 51 layers), the 8,192 setting, seed 1, the 15 sometimes-solved tasks (Colab G4); against unedited (9 to 13, 65 of 90), the correlation edits (57 of 88) and the 64-neuron causal edit (EXP-161a, 9 of 15) (D-227)

- **Status:** OK, recorded automatically by tools/colab_batch.py. Read critically in RESEARCH_LOG.md.
- **Where:** Google Colab, G4, official stack (the Kaggle notebook's code, with input and working folders under /content).
- **Model:** gemma-4-31b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** ['fastapi_13537', 'fastapi_14463', 'fastapi_15589', 'fastapi_5624', 'fastapi_9425', 'fastapi_9753', 'rich_2725', 'rich_3043', 'rich_3063', 'rich_3067', 'rich_3278', 'rich_3470', 'rich_3480', 'rich_4077', 'rich_4079']. Unfinished tasks count as unresolved.
- **Wall time:** about 2440 s.

## Results
Colab tests (not official) 6/15 resolved of 15 finished; {'patch failed tests': 9, 'resolved': 6}; 38/83 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

**The resolved count above is not the official score.** Colab's test environment does not match the Kaggle image the official scorer uses (25 of 58 reference fixes pass there). Agent behaviour (tool calls, edits, loops, patches) is valid; whether a patch solves its task is decided by tools/kaggle_verify.py on a Kaggle CPU notebook and recorded in `kaggle_verify.json`.

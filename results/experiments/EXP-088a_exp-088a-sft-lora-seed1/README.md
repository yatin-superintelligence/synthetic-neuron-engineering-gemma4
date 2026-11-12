# EXP-088a: official E2B, EXP-088 SFT LoRA, seed 1 (Colab L4)

- **Status:** OK, recorded automatically by tools/colab_batch.py. Read critically in RESEARCH_LOG.md.
- **Where:** Google Colab, L4, official stack (the Kaggle notebook's code, with input and working folders under /content).
- **Model:** gemma-4-e2b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** small. Unfinished tasks count as unresolved.
- **Wall time:** about 4167 s.

## Results
Colab tests (not official) 1/56 resolved of 56 finished; {'no patch': 17, 'error: Sandbox execution error: litellm.Context': 17, 'patch failed tests': 14, 'error: Agent exceeded tool call budget (100 cal': 5, 'resolved': 1, 'error: Agent completed execution without callin': 1, 'error: Agent exceeded maximum allowed LLM turns': 1}; 34/211 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

**The resolved count above is not the official score.** Colab's test environment does not match the Kaggle image the official scorer uses (25 of 58 reference fixes pass there). Agent behaviour (tool calls, edits, loops, patches) is valid; whether a patch solves its task is decided by tools/kaggle_verify.py on a Kaggle CPU notebook and recorded in `kaggle_verify.json`.

# EXP-013K: official E4B baseline (hard stop 2000 s to fit the last 0.91 h of weekly quota)

- **Status:** OK, recorded automatically by tools/kaggle_batch.py. Read critically in RESEARCH_LOG.md.
- **Model:** gemma-4-e4b-it-qat-w4a16-ct
- **Change from the default:** see `config_override.json`.
- **Budgets:** 60 minutes, 100 tool calls, 300 s per command.
- **Tasks:** the 58 public small-fix tasks. Unfinished tasks count as unresolved.
- **Wall time:** about 2046 s.

## Results
6/58 resolved of 58 finished; {'patch failed tests': 43, 'error: Sandbox execution error: litellm.Context': 6, 'resolved': 6, 'no patch': 3}; 68/126 edits applied

Numbers are in `summary.json`, from `tools/trace_summary.py`. Traces are not stored in this repository.

## Reading

Compared with EXP-044a: E2B, same seed and settings, same 58 tasks. All 58 tasks finished before the 2000 s stop, so nothing was lost to it. Kaggle let the run finish at 6.21 of the 6 h quota.

| Of 58 tasks unless stated | E2B | E4B |
|---|---|---|
| Solved | 5 | 6 |
| Tasks with a loop | 16 | 0 |
| Responses of at least 15,000 tokens | 6 | 0 |
| Context errors | 13 | 6 |
| Tasks that never applied an edit | 18 | 6 |
| Failed edit calls | 51 of 110 | 58 of 126 |
| Mean tool calls per task | 8.4 | 12.4 |

3 tasks were solved by both, 3 by E4B only and 2 by E2B only, so one seed shows no gain in solved tasks. The zero loop count was checked, because a constant reading can mean a broken instrument: thought text is present, and 5 of E4B's 886 model calls repeated a sentence at least 8 times, but none scored above 0.37 on repetition, against the threshold of 0.5 and scores above 0.9 for E2B's worst loops. See `analysis/e4b_check.py`.

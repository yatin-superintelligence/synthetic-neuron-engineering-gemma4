# EXP-074 (partial): official 31B, no edit, seed 1, Colab A100, stopped at 16 of 58 tasks

- **Status:** stopped by hand at 2026-11-08 00:12 UTC, 61 minutes after launch, with 16 of 58 tasks finished.
- **Why stopped:** at 14 tasks in 46 minutes the run would have hit its 3-hour limit with about 50 tasks done, at about 20 units.
  31B's weights leave a 40 GB A100 too little memory for 8 long contexts at once (37.7 of 41 GB used, GPU at 100%). The full
  baseline is rerun on a G4 (96 GB). The finished tasks were saved before the stop (artifacts_local/colab/exp-074-partial).
- **What it can show:** an early, partial reading of 31B's solve rate on the tasks that finish first (shorter tasks finish first, so
  this subset is not a random sample of the 58). Scored by the official verification on Kaggle (kaggle_verify.json, when present).

## Reading

5 of 16 resolved by the official verification on Kaggle (fastapi_14463, fastapi_14492, fastapi_14786, fastapi_14794, rich_4077).
Unedited E2B on the same 16 tasks: 1 of 16 at seed 1 and 1 of 16 at seed 2 (fastapi_14786 both times).

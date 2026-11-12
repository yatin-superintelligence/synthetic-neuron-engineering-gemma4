# EXP-083: what Gemma 4 31B would do at E4B's own agent states

- **Status:** OK. Colab G4, official stack (31B W4A16), 2026-11-08 03:46 to 05:48 UTC, 2.03 h, about 20 units.
- **Procedure:** EXP-079's, unchanged (experiments/scripts/exp079_teacher_decisions.py), on 1,751 states of two unedited E4B runs
  (EXP-073a, EXP-073c): 4 samples of 31B per state, 7,004 teacher actions. Labels: artifacts_local/EXP-083/teacher_labels.jsonl
  (ids, actions, token counts). The summary's field names say "e2b" because the script was written for E2B; here they mean E4B.

## Results

31B's majority action equals E4B's at 744 of 1,751 states (42%; E2B: 596 of 1,745, 34%). 31B's actions: run_command 2,783 of
7,004 samples (40%), read_file 2,084 (30%), text answer 583 (8%), edit_file 438 (6%), write_file 345 (5%), semantic search 266 (4%).
E4B's actions at these states: read_file 643 (37%), run_command 331 (19%), edit_file 232 (13%), semantic search 219 (13%).
Confident disagreements (3 or 4 of 4 samples): read_file -> run_command 210, search -> run_command 96, edit_file -> read_file 64,
edit_file -> run_command 40, search -> read_file 34, run_command -> write_file 32, run_command -> read_file 31.
The same pattern as for E2B (EXP-079): 31B prefers the shell and reading before editing; E4B is closer to 31B than E2B is.

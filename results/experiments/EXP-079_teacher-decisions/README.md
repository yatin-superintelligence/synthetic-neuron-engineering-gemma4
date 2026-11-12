# EXP-079: what Gemma 4 31B would do at E2B's own agent states

- **Status:** OK. Colab G4, official stack (vLLM 0.19.1, 31B W4A16), 2026-11-07 23:43 to 2026-11-08 01:24 UTC, 1.68 h, about 17 units.
- **Script:** experiments/scripts/exp079_teacher_decisions.py (snapshot here). Labels: artifacts_local/EXP-079/teacher_labels.jsonl
  (ids, actions and token counts only; no task text). Design fixed before the run is in the script's docstring.
- **States:** 1,745 recorded E2B agent states from the three unedited base runs (exp029, exp037, exp-044a): 1,597 tool calls and 148
  text answers by the main agent. Prompts: median 10,050 tokens, at most 21,998 after cutting old history. 31B continued each state
  with its own thinking and action, 4 samples each (temperature 1.0, top_p 0.95, top_k 64), so 6,980 teacher actions.

## Results

| action | 31B samples (of 6,980) | E2B states (of 1,745) |
|---|---|---|
| run_command | 2,526 (36%) | 173 (10%) |
| read_file | 1,915 (27%) | 564 (32%) |
| edit_file | 651 (9%) | 328 (19%) |
| STOP (text answer, no call) | 572 (8%) | 148 (8%) |
| submit_patch | 367 (5%) | 128 (7%) |
| search_similar_code | 345 (5%) | 288 (17%) |
| write_file | 326 (5%) | 61 (3%) |
| UNDECIDED (3,072 tokens without a decision) | 216 (3%) | - |
| code_analyzer_agent (sub-agent) | 8 (0.1%) | 43 (2%) |

31B's majority action equals E2B's at 596 of 1,745 states (34%). Confident disagreements (3 or 4 of 4 samples on one other action):
read_file -> run_command 194, search_similar_code -> run_command 123, edit_file -> read_file 83, edit_file -> run_command 67,
search_similar_code -> read_file 44, submit_patch -> run_command 27, read_file -> edit_file 25, code_analyzer_agent -> run_command 23,
STOP -> submit_patch 20; smaller ones in results.json.

Reading: 31B works through the shell (grep, cat, tests) where E2B uses file reads and semantic search; it reads before editing where
E2B edits at once (83 states); and it runs a command before submitting where E2B submits (27 states). It almost never delegates to
the code-analysis sub-agent. These are the decisions EXP-080 turns into synthetic neurons for E2B. 31B is the stronger agent on the
tasks measured so far (EXP-074 partial: 5 of 16 against E2B's 1 of 16 on the same tasks).

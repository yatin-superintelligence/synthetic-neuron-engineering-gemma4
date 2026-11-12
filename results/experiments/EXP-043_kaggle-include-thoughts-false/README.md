# EXP-043: official E2B run with include_thoughts false

- **Status:** OK, stopped by its 25-minute limit with 55 of 58 tasks finished. The 3 unfinished tasks count as unresolved.
- **Why:** EXP-037 lost 15 tasks to context errors, 10 of them after a loop. With include_thoughts false, thoughts still
  run on each turn but are removed from later history, so a loop cannot fill the context through its thinking.
- **Change from EXP-037:** only `thinking_config.include_thoughts: false`. Same model, tasks, budgets and prompt.
- **Stack:** private Kaggle notebook sne-exp043-e2b-include-thoughts-false, L4 GPUs, vLLM, W4A16, swegemma 0.2.11.
- **Cost:** 1929 s of notebook time, 1.07 h of quota. The run passed the 6 h quota line without being stopped.

## Results
Resolved 5/58 (8.6%): fastapi_14301, fastapi_14786, requests_6644, rich_2943, rich_3905. EXP-037 resolved 2/58.
Outcomes on the 55 finished: patch failed tests 34, no patch 10, context error 6, resolved 5.
Mean 12.2 tool calls, against 8.3. Numbers are in `summary.json`.

## Reading
- Context errors fell from 15 to 6, which is the predicted effect.
- Both tasks resolved by EXP-037 also resolved here, so the discordant pairs are 3 to 0, McNemar p = 0.25. One run cannot
  separate this from noise. The noise floor is unmeasured, and EXP-036 resolved 4 with a stricter budget. Do not claim this yet.
- 146 of 195 edit_file calls failed (75%), against 56% in EXP-037. Removing thoughts from history may cost the model its
  working notes, or the count may rise only because runs last longer and repeat failed calls: 69 failures repeat the previous
  call. Edit failures got worse while resolved tasks did not, so apply rate alone is not a reliable proxy for success.
- The loop count here is not comparable with EXP-037. The traces no longer hold thoughts, so the detector sees only visible
  output and undercounts thinking loops.

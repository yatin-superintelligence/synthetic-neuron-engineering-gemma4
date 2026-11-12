# EXP-057: how many parallel slots finish a batch of long agent contexts fastest?

- **Status:** OK. Stopped by decision after the F16 2-slot run; the remaining F16 arms were not needed for the choice.
- **Why:** slot count was tuned on short prompts and carried over untested to long contexts, where every replay spends its time.
- **Hypothesis:** fewer slots finish sooner. Supported.
- **Method:** the same 6 real loop contexts, 83266 prompt tokens in all, each generating up to 256 tokens, batch 512, full wall time.
- **Outputs:** `throughput.json`. The first attempt, at batch 1024, lost the GPU device and is in `experiments/_failed`.

## Results
| Precision | Slots | Cache | Wall (s) | s per state |
|---|---|---|---|---|
| q4_0 | 1 | separate | 281.5 | 46.9 |
| q4_0 | 2 | separate | 533.8 | 89.0 |
| q4_0 | 3 | separate | 511.0 | 85.2 |
| q4_0 | 4 | separate | 502.4 | 83.7 |
| q4_0 | 6 | separate | 506.7 | 84.5 |
| q4_0 | 3 | unified | 522.6 | 87.1 |
| q4_0 | 6 | unified | 555.3 | 92.6 |
| f16 | 1 | separate | 363.7 | 60.6 |
| f16 | 2 | separate | 661.8 | 110.3 |

## Reading
- **One slot is close to twice as fast:** 46.9 s per state against 84 to 93 on Q4_0, and 60.6 against 110.3 on F16.
- **It is a step, not a slope.** 2, 3, 4 and 6 slots all land between 84 and 89 s, so the loss comes from having more than one slot
  at all, not from each extra slot. The cause inside the engine is not established.
- **A unified cache did not help:** 87.1 s at 3 slots and 92.6 s at 6.
- **Limit of this test.** It generates 256 tokens per state. Long generations favour parallel slots more, because parallel decoding
  does add throughput. At 19 tokens/s on one slot and about 32 across six, a 1200-token response still leaves one slot level or
  ahead once prompt time is counted. So one slot is the choice for screens; a run dominated by generation should be rechecked.
- No crash occurred at batch 512 in any arm, including 20000-token contexts.
- Decision: replays use `SLOTS = 1`. Real-task Kaggle runs are not affected.

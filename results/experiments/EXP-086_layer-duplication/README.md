# EXP-086: layer-duplication

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T13:39:05.417108Z / 2026-11-08T16:34:47.855866Z
- **Wall time:** 175.7 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-044a
- **Command:** `experiments/scripts/exp086_layer_duplication.py`

## Why this experiment
The brief asks for more layers as well as more neurons. Re-running a block of E2B's own layers is the training-free way to add depth; it has not been tested here (idea I6).

## Hypothesis (stated before running)
Some block, run twice, raises E2B's MBPP+ pass count on the held-out test half against the unedited model (exact McNemar p < 0.05) and against a control block of the same size; most blocks lower it.

## Configuration
```json
{
  "blocks": [
    [
      1,
      2
    ],
    [
      3,
      2
    ],
    [
      5,
      2
    ],
    [
      7,
      2
    ],
    [
      9,
      2
    ],
    [
      11,
      2
    ],
    [
      15,
      2
    ],
    [
      17,
      2
    ],
    [
      19,
      2
    ],
    [
      21,
      2
    ],
    [
      23,
      2
    ],
    [
      25,
      2
    ],
    [
      27,
      2
    ],
    [
      29,
      2
    ],
    [
      31,
      2
    ],
    [
      33,
      2
    ],
    [
      3,
      4
    ],
    [
      7,
      4
    ],
    [
      17,
      4
    ],
    [
      23,
      4
    ],
    [
      29,
      4
    ]
  ],
  "scan": "MBPP+ dev 189, answers capped at 4,096 new tokens (D-057)",
  "confirm": "MBPP+ test 189, W4A16 and bf16",
  "max_tokens": 8192,
  "thinking": false,
  "engine": "vLLM 0.19.1 (official stack), enforce_eager, greedy",
  "gate": "McNemar p < 0.05 on test at W4A16, not lower at bf16"
}
```

## Results
```json
{
  "scan": {
    "official": [
      110,
      {
        "pass": 110,
        "wrong_answer": 49,
        "wrong_or_missing_entry_point": 9,
        "over_cap": 3,
        "timeout": 1,
        "runtime_error": 17
      }
    ],
    "identity": [
      111,
      {
        "pass": 111,
        "wrong_answer": 45,
        "wrong_or_missing_entry_point": 10,
        "timeout": 1,
        "runtime_error": 16,
        "over_cap": 5,
        "other_fail": 1
      }
    ],
    "dup_s1_k2": [
      0,
      {
        "over_cap": 77,
        "wrong_or_missing_entry_point": 50,
        "no_code_block": 60,
        "syntax_error": 2
      }
    ],
    "dup_s3_k2": [
      0,
      {
        "loop": 57,
        "no_code_block": 67,
        "wrong_or_missing_entry_point": 62,
        "syntax_error": 3
      }
    ],
    "dup_s5_k2": [
      0,
      {
        "loop": 84,
        "wrong_or_missing_entry_point": 56,
        "syntax_error": 32,
        "truncated": 16,
        "name_or_import_error": 1
      }
    ],
    "dup_s7_k2": [
      0,
      {
        "loop": 147,
        "truncated": 8,
        "wrong_or_missing_entry_point": 34
      }
    ],
    "dup_s9_k2": [
      0,
      {
        "loop": 116,
        "wrong_or_missing_entry_point": 31,
        "no_code_block": 22,
        "truncated": 18,
        "name_or_import_error": 1,
        "syntax_error": 1
      }
    ],
    "dup_s11_k2": [
      0,
      {
        "no_code_block": 99,
        "loop": 23,
        "truncated": 58,
        "wrong_or_missing_entry_point": 9
      }
    ],
    "dup_s15_k2": [
      11,
      {
        "truncated": 37,
        "loop": 140,
        "pass": 11,
        "wrong_or_missing_entry_point": 1
      }
    ],
    "dup_s17_k2": [
      73,
      {
        "truncated": 27,
        "pass": 73,
        "loop": 23,
        "wrong_or_missing_entry_point": 12,
        "wrong_answer": 25,
        "syntax_error": 16,
        "runtime_error": 11,
        "name_or_import_error": 1,
        "other_fail": 1
      }
    ],
    "dup_s19_k2": [
      6,
      {
        "truncated": 91,
        "loop": 89,
        "wrong_or_missing_entry_point": 1,
        "wrong_answer": 2,
        "pass": 6
      }
    ],
    "dup_s21_k2": [
      42,
      {
        "truncated": 70,
        "wrong_answer": 22,
        "syntax_error": 15,
        "loop": 31,
        "pass": 42,
        "runtime_error": 6,
        "wrong_or_missing_entry_point": 3
      }
    ],
    "dup_s23_k2": [
      0,
      {
        "loop": 162,
        "truncated": 26,
        "wrong_or_missing_entry_point": 1
      }
    ],
    "dup_s25_k2": [
      89,
      {
        "pass": 89,
        "runtime_error": 20,
        "wrong_answer": 47,
        "loop": 8,
        "wrong_or_missing_entry_point": 7,
        "truncated": 6,
        "timeout": 1,
        "syntax_error": 9,
        "name_or_import_error": 2
      }
    ],
    "dup_s27_k2": [
      92,
      {
        "pass": 92,
        "wrong_answer": 47,
        "wrong_or_missing_entry_point": 19,
        "syntax_error": 5,
        "runtime_error": 20,
        "loop": 2,
        "name_or_import_error": 1,
        "truncated": 3
      }
    ],
    "dup_s29_k2": [
      58,
      {
        "runtime_error": 13,
        "wrong_answer": 26,
        "loop": 37,
        "wrong_or_missing_entry_point": 18,
        "name_or_import_error": 8,
        "pass": 58,
        "truncated": 18,
        "syntax_error": 10,
        "timeout": 1
      }
    ],
    "dup_s31_k2": [
      97,
      {
        "truncated": 1,
        "pass": 97,
        "wrong_answer": 44,
        "wrong_or_missing_entry_point": 17,
        "runtime_error": 16,
        "timeout": 2,
        "loop": 4,
        "syntax_error": 7,
        "name_or_import_error": 1
      }
    ],
    "dup_s33_k2": [
      0,
      {
        "truncated": 157,
        "loop": 30,
        "wrong_or_missing_entry_point": 2
      }
    ],
    "dup_s3_k4": [
      0,
      {
        "loop": 41,
        "no_code_block": 83,
        "wrong_or_missing_entry_point": 58,
        "syntax_error": 4,
        "truncated": 3
      }
    ],
    "dup_s7_k4": [
      0,
      {
        "loop": 154,
        "truncated": 10,
        "wrong_or_missing_entry_point": 3,
        "no_code_block": 22
      }
    ],
    "dup_s17_k4": [
      0,
      {
        "truncated": 50,
        "loop": 139
      }
    ],
    "dup_s23_k4": [
      0,
      {
        "loop": 185,
        "wrong_or_missing_entry_point": 4
      }
    ],
    "dup_s29_k4": [
      14,
      {
        "loop": 82,
        "truncated": 48,
        "pass": 14,
        "wrong_answer": 9,
        "syntax_error": 11,
        "wrong_or_missing_entry_point": 13,
        "runtime_error": 6,
        "name_or_import_error": 6
      }
    ]
  },
  "selected": "dup_s31_k2",
  "control": "dup_s5_k2",
  "test_w4a16": {
    "official": 121,
    "dup_s31_k2": 109,
    "dup_s5_k2": 0
  },
  "test_bf16": {},
  "stats": {
    "best_only": 15,
    "official_only": 27,
    "p": 0.0884,
    "best_vs_control": {
      "best_only": 109,
      "control_only": 0,
      "p": 0.0
    },
    "bf16": {
      "error": "RuntimeError('vLLM exited while loading: APIServer pid=108525)     async with build_async_engine_client_from_engine_args(\\n(APIServer pid=108525)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n(APIServer pid=108525)   File \"/root/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/contextlib.py\", line 210, in __aenter__\\n(APIServer pid=108525)     return await anext(self.gen)\\n(APIServer pid=108525)            ^^^^^^^^^^^^^^^^^^^^^\\n(APIServer pid=108525)   File \""
    }
  },
  "gate_passed": false,
  "headline": "selected dup_s31_k2: dev 97 vs official 110 of 189; test W4A16 109 vs 121 of 189 (McNemar p = 0.0884); gate not passed",
  "system_monitor": {
    "samples": 215
  }
}
```

## Files
- `script_snapshot.py` exact script that produced this folder (+ `uncommitted.diff` if any)
- `run.log` full console log
- `config.json` config + environment provenance
- `results.json` metrics
- `sysmon.jsonl` RAM/GPU/battery/thermal samples every 60 s
- other `*.json` raw outputs (generations, per-problem outcomes)

# EXP-002: harness-validation

- **Status:** OK
- **Category:** infrastructure
- **Started / finished (UTC):** 2026-11-05T19:17:14.844834Z / 2026-11-05T19:19:10.955989Z
- **Wall time:** 1.9 min
- **Commit:** `1970dc14b424a252a8e4cc5848ebfd2cc451fbc7` (dirty=False)
- **Depends on:** -
- **Command:** `experiments/scripts/exp002_harness_validation.py`

## Why this experiment
Every later number depends on prompt assembly, code extraction and test execution being correct. A reviewer should be able to see that the harness passes reference solutions and rejects known-buggy ones before trusting any model score.

## Hypothesis (stated before running)
Canonical solutions pass 100% on all suites; buggy HumanEvalFix solutions fail 100%.

## Configuration
```json
{
  "suites": [
    "mbpp_dev",
    "mbpp_test",
    "humaneval",
    "humanevalfix"
  ],
  "timeout_s": 30
}
```

## Results
```json
{
  "mbpp_dev": {
    "n": 189,
    "canonical_pass": 189,
    "failures": []
  },
  "mbpp_test": {
    "n": 188,
    "canonical_pass": 188,
    "failures": []
  },
  "humaneval": {
    "n": 163,
    "canonical_pass": 163,
    "failures": []
  },
  "humanevalfix": {
    "n": 164,
    "canonical_pass": 164,
    "failures": []
  },
  "humanevalfix_buggy_hidden": {
    "n": 164,
    "buggy_unexpectedly_pass": []
  },
  "humanevalfix_canonical_visible": {
    "fail": []
  },
  "humanevalfix_buggy_visible_pass": 20,
  "extraction_unit_tests": [
    true,
    true,
    true,
    true
  ],
  "headline": "mbpp_dev: 189/189 | mbpp_test: 188/188 | humaneval: 163/163 | humanevalfix: 164/164"
}
```

## Files
- `run.log` full console log
- `config.json` config + environment provenance
- `results.json` metrics
- other `*.json` raw outputs (generations, per-problem outcomes)

# EXP-135: 12b-loop-breaker-neuron

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-10T23:22:04.617108Z / 2026-11-10T23:23:08.220173Z
- **Wall time:** 1.1 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-014d, EXP-051, EXP-132
- **Command:** `experiments/scripts/exp135_12b_loop_breaker.py`

## Why this experiment
12B's looping thought runs to the thinking cap and the run dies of a context overflow. A neuron that reads 'a sentence was just repeated' and writes the end-of-thought token ends the thought at the loop's start.

## Hypothesis (stated before running)
The read-out separates loop sentence ends from normal sentence ends on held-out tasks (99th percentile at normal sentence ends under 0.1); the agent test measures loops, overflows and tasks solved.

## Configuration
```json
{
  "model": "google/gemma-4-12B-it-qat-q4_0-unquantized",
  "ctx_tokens": 3000,
  "resp_tokens": 3000,
  "lambdas": [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0
  ],
  "split": "dev",
  "held_out_fraction": "1/3 of dev tasks, seed 135",
  "selection_rule": "max held-out pos_mean with hard-negative p99 < 0.1"
}
```

## Results
```json
{
  "fit": {
    "positives": 8008,
    "negatives": 8951,
    "hard": 1856,
    "held_positives": 1820,
    "held_hard": 654,
    "median_gap": 17.438,
    "c": 9.0,
    "layer": 47
  },
  "adapters": [],
  "headline": "no ridge strength passed the selection rule (gap 17.44); no neuron built",
  "system_monitor": {
    "samples": 2
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

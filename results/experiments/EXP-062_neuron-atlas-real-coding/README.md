# EXP-062: neuron-atlas-real-coding

- **Status:** OK
- **Category:** localization
- **Started / finished (UTC):** 2026-11-06T17:25:13.526997Z / 2026-11-06T18:01:34.720811Z
- **Wall time:** 36.4 min
- **Commit:** `b9f3b10144c0b4b7611aad4c5af0a4c67c7e372d` (dirty=True)
- **Depends on:** EXP-041
- **Command:** `experiments/scripts/exp062_neuron_atlas.py`

## Why this experiment
The field needs measured maps, not guesses. No one has mapped which Gemma neurons respond to which part of agentic coding, and every neuron edit should start from such a map.

## Hypothesis (stated before running)
No directional claim: this is a measurement. Reported as found, including if nothing replicates.

## Configuration
```json
{
  "ctx_tokens": 3000,
  "d_min": 1.0,
  "clusters": 16,
  "classes": [
    "system",
    "issue",
    "thought",
    "call_syntax",
    "arg_old",
    "arg_new",
    "arg_command",
    "arg_path",
    "tool_response",
    "model_text",
    "keyword",
    "identifier",
    "punctuation",
    "whitespace",
    "number",
    "other"
  ],
  "split": "dev",
  "half_seed": 62
}
```

## Results
```json
{
  "classes": [
    {
      "cls": "system",
      "tokens": 43232,
      "spearman_between_halves": 0.975,
      "n_replicated": 530
    },
    {
      "cls": "issue",
      "tokens": 48684,
      "spearman_between_halves": 0.964,
      "n_replicated": 143
    },
    {
      "cls": "thought",
      "tokens": 134261,
      "spearman_between_halves": 0.92,
      "n_replicated": 32
    },
    {
      "cls": "call_syntax",
      "tokens": 6634,
      "spearman_between_halves": 0.983,
      "n_replicated": 2349
    },
    {
      "cls": "arg_old",
      "tokens": 13973,
      "spearman_between_halves": 0.849,
      "n_replicated": 942
    },
    {
      "cls": "arg_new",
      "tokens": 23904,
      "spearman_between_halves": 0.758,
      "n_replicated": 26
    },
    {
      "cls": "arg_command",
      "tokens": 31,
      "spearman_between_halves": NaN,
      "n_replicated": 0
    },
    {
      "cls": "arg_path",
      "tokens": 4595,
      "spearman_between_halves": 0.607,
      "n_replicated": 0
    },
    {
      "cls": "tool_response",
      "tokens": 191161,
      "spearman_between_halves": 0.917,
      "n_replicated": 20
    },
    {
      "cls": "model_text",
      "tokens": 16310,
      "spearman_between_halves": 0.611,
      "n_replicated": 2
    },
    {
      "cls": "keyword",
      "tokens": 14532,
      "spearman_between_halves": 0.909,
      "n_replicated": 19
    },
    {
      "cls": "identifier",
      "tokens": 110408,
      "spearman_between_halves": 0.946,
      "n_replicated": 1
    },
    {
      "cls": "punctuation",
      "tokens": 68069,
      "spearman_between_halves": 0.947,
      "n_replicated": 5
    },
    {
      "cls": "whitespace",
      "tokens": 18592,
      "spearman_between_halves": 0.905,
      "n_replicated": 263
    },
    {
      "cls": "number",
      "tokens": 15687,
      "spearman_between_halves": 0.799,
      "n_replicated": 12
    },
    {
      "cls": "other",
      "tokens": 1750,
      "spearman_between_halves": 0.755,
      "n_replicated": 184
    }
  ],
  "groups": 16,
  "responding": 8598,
  "total": 337920,
  "headline": "8598 of 337920 neurons have a replicated class response; largest classes call_syntax 2349, arg_old 942, system 530, whitespace 263",
  "system_monitor": {
    "samples": 31,
    "ram_free_gb": {
      "min": 2.74,
      "max": 20.46,
      "mean": 7.39
    },
    "gpu_shared_gb": {
      "min": 1.25,
      "max": 12.78,
      "mean": 8.44
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 84.71
    },
    "cpu_package_w": {
      "min": 15.4,
      "max": 44.8,
      "mean": 29.25
    },
    "cpu_core_w": {
      "min": 6.6,
      "max": 14.6,
      "mean": 8.96
    },
    "igpu_w": {
      "min": 0,
      "max": 29.2,
      "mean": 14.64
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 43,
      "mean": 9.37
    },
    "cpu_perf_pct": {
      "min": 106.1,
      "max": 175.5,
      "mean": 147.59
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "passive_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "throttle_reasons": {
      "min": 0,
      "max": 0,
      "mean": 0.0
    },
    "cpu_freq_mhz": {
      "min": 1647,
      "max": 1913,
      "mean": 1788.74
    },
    "battery_discharge_mw": {
      "min": 0,
      "max": 0,
      "mean": 0.0
    },
    "battery_charge_mw": {
      "min": 0,
      "max": 0,
      "mean": 0.0
    }
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

## Reading
- **A clear map exists, and it replicates.** Of 337,920 neurons, 8,598 (2.5%) respond to a class at d of at least 1 in both task halves. Selectivity
  across all neurons correlates 0.76 to 0.98 between halves for classes with enough tokens.
- **What the model is doing sits mid-network.** Neurons tuned to activity (tool syntax, old_string, system text, issue) number under 20 per
  layer up to layer 12, then 53, 358, 933, 688, 564 and 326 across layers 13 to 18, then decline. Layer 15 is where the MLP doubles in width and
  key-value sharing starts, so these are counts: 5.8% of layer 14 and 7.6% of layer 15.
- **What kind of token it is sits at the two ends.** Lexical tuning peaks at layer 0 (136 neurons) and layer 34 (38), with almost none at layers 14 to 20.
- **Copying looks special.** old_string has 942 replicated neurons, new_string 26, though new_string has more tokens. Caution: the model writes
  old_string after new_string in nearly every call, so this class also means late in a tool call, and it may reflect copying, position
  or both.
- **The loop direction peaked at layers 14 and 15 in EXP-040,** at the same place. That is a coincidence of location until tested.
- **Limits.** Token identity is not controlled, so some syntax neurons may simply respond to `{`, `:` and the like. This is 101 states from 27 tasks in 3
  repositories. Reading group writes through the vocabulary gave mostly noise, so the group labels come from responses alone.
- **This shows response, not responsibility.** EXP-063 switches groups off.

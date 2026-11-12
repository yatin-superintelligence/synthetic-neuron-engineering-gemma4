# EXP-100: fix-neuron-map

- **Status:** OK
- **Category:** localization
- **Started / finished (UTC):** 2026-11-09T08:57:53.351476Z / 2026-11-09T09:16:24.522562Z
- **Wall time:** 18.5 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-092
- **Command:** `experiments/scripts/exp100_fix_neuron_map.py`

## Why this experiment
Map which of E2B's MLP neurons are more active when it writes a correct fix than a wrong one, and edit the model through them.

## Hypothesis (stated before running)
Amplifying the neurons most tied to correct fixes (or damping those tied to wrong ones) raises correct bench answers by at least half against the unedited model, beyond random neurons scaled the same way.

## Configuration
```json
{
  "cap_per_layer": 128,
  "edits": [
    "fix_top64_b15",
    "fix_top64_b25",
    "fix_top256_b15",
    "fix_top256_b25",
    "wrong_top256_b05",
    "rnd256_b25",
    "rnd256_b05"
  ]
}
```

## Results
```json
{
  "map": {
    "answers": 496,
    "correct": 96,
    "split_half_correlation_top2000": 0.9883,
    "strong_neurons_per_layer_abs_t_over_4": {
      "0": 1589,
      "1": 1803,
      "2": 2021,
      "3": 2169,
      "4": 1924,
      "5": 2265,
      "6": 2563,
      "7": 2795,
      "8": 3099,
      "9": 2905,
      "10": 3156,
      "11": 3036,
      "12": 2889,
      "13": 3010,
      "14": 3353,
      "15": 6912,
      "16": 6918,
      "17": 6672,
      "18": 6790,
      "19": 6199,
      "20": 6123,
      "21": 6060,
      "22": 6044,
      "23": 5961,
      "24": 5652,
      "25": 5395,
      "26": 5484,
      "27": 5530,
      "28": 5621,
      "29": 5331,
      "30": 5386,
      "31": 5309,
      "32": 5249,
      "33": 5161,
      "34": 4249
    },
    "fix256_per_layer": {
      "2": 1,
      "7": 1,
      "8": 3,
      "9": 2,
      "10": 5,
      "11": 5,
      "12": 3,
      "13": 5,
      "14": 29,
      "15": 63,
      "16": 37,
      "17": 19,
      "18": 20,
      "19": 9,
      "20": 14,
      "21": 6,
      "22": 5,
      "23": 4,
      "24": 8,
      "25": 1,
      "27": 2,
      "28": 3,
      "29": 2,
      "30": 2,
      "31": 3,
      "32": 1,
      "33": 3
    },
    "wrong256_per_layer": {
      "7": 2,
      "8": 4,
      "9": 1,
      "10": 3,
      "11": 2,
      "12": 4,
      "13": 6,
      "14": 26,
      "15": 61,
      "16": 37,
      "17": 22,
      "18": 16,
      "19": 12,
      "20": 14,
      "21": 6,
      "22": 6,
      "23": 12,
      "24": 4,
      "25": 4,
      "27": 2,
      "28": 2,
      "29": 2,
      "30": 1,
      "31": 2,
      "32": 4,
      "33": 1
    },
    "top10": [
      {
        "layer": 11,
        "neuron": 974,
        "t": 21.13
      },
      {
        "layer": 15,
        "neuron": 8891,
        "t": 20.27
      },
      {
        "layer": 14,
        "neuron": 5712,
        "t": 20.1
      },
      {
        "layer": 15,
        "neuron": 680,
        "t": 19.95
      },
      {
        "layer": 16,
        "neuron": 11560,
        "t": 19.8
      },
      {
        "layer": 15,
        "neuron": 2386,
        "t": 19.51
      },
      {
        "layer": 14,
        "neuron": 4955,
        "t": 19.47
      },
      {
        "layer": 16,
        "neuron": 8760,
        "t": 19.12
      },
      {
        "layer": 16,
        "neuron": 461,
        "t": 19.05
      },
      {
        "layer": 15,
        "neuron": 4817,
        "t": 18.74
      }
    ],
    "bottom10": [
      {
        "layer": 14,
        "neuron": 4205,
        "t": -19.84
      },
      {
        "layer": 16,
        "neuron": 9484,
        "t": -19.66
      },
      {
        "layer": 15,
        "neuron": 10752,
        "t": -19.62
      },
      {
        "layer": 22,
        "neuron": 3688,
        "t": -19.58
      },
      {
        "layer": 15,
        "neuron": 2713,
        "t": -19.58
      },
      {
        "layer": 16,
        "neuron": 2103,
        "t": -19.11
      },
      {
        "layer": 16,
        "neuron": 7671,
        "t": -18.94
      },
      {
        "layer": 16,
        "neuron": 7476,
        "t": -18.84
      },
      {
        "layer": 15,
        "neuron": 1592,
        "t": -18.72
      },
      {
        "layer": 16,
        "neuron": 6110,
        "t": -18.7
      }
    ]
  },
  "adapters": [
    "fix_top256_b15",
    "fix_top256_b25",
    "fix_top64_b15",
    "fix_top64_b25",
    "rnd256_b05",
    "rnd256_b25",
    "wrong_top256_b05"
  ],
  "headline": "fix-neuron map: 96 correct of 496 answers; split-half t correlation 0.99 over the top 2,000 neurons; strong neurons by layer {0: 1589, 1: 1803, 2: 2021, 3: 2169, 4: 1924, 5: 2265, 6: 2563, 7: 2795, 8: 3099, 9: 2905, 10: 3156, 11: 3036, 12: 2889, 13: 3010, 14: 3353, 15: 6912, 16: 6918, 17: 6672, 18: 6790, 19: 6199, 20: 6123, 21: 6060, 22: 6044, 23: 5961, 24: 5652, 25: 5395, 26: 5484, 27: 5530, 28: 5621, 29: 5331, 30: 5386, 31: 5309, 32: 5249, 33: 5161, 34: 4249}",
  "system_monitor": {
    "samples": 19
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

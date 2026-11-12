# EXP-085: qwen-transplant

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-08T06:52:42.343327Z / 2026-11-08T06:59:49.437432Z
- **Wall time:** 7.1 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-077
- **Command:** `experiments/scripts/exp085_qwen_transplant.py`

## Why this experiment
Add capacity to E2B from another model family (Qwen3.5-4B), fixing the write side that failed for E4B (EXP-077).

## Hypothesis (stated before running)
Transplanted Qwen neurons with writes fitted in E2B's space lower held-out code loss below both the unedited model and random-Qwen-neuron controls.

## Configuration
```json
{
  "donor": "Qwen/Qwen3.5-4B",
  "k": 64,
  "layer_range": [
    8,
    28
  ],
  "scales": [
    0.25,
    0.5,
    1.0,
    2.0
  ],
  "chunk": 2400,
  "positions_per_chunk": 24,
  "fit_repos": [
    "fastapi",
    "requests"
  ],
  "test_repo": "rich",
  "n_fit": 360,
  "n_test": 120
}
```

## Results
```json
{
  "layers": {
    "25": {
      "donor_layer": 30,
      "cka": 0.741,
      "read_map_ridge": 1.0,
      "read_map_r2_fit": 0.486,
      "read_map_r2_cv_requests": 0.196,
      "read_map_r2_heldout": 0.219,
      "transplant": {
        "read_fidelity_median": 0.448,
        "read_fidelity_p25": 0.32,
        "write_fit_r2": 0.0173,
        "write_ridge": 0.1,
        "write_r2_cv": -0.0004,
        "neurons": [
          2720,
          623,
          2381,
          281,
          1886,
          444,
          3645,
          4602,
          575,
          3910,
          815,
          3566,
          156,
          1356,
          2560,
          2638
        ]
      },
      "control_random": {
        "read_fidelity_median": 0.387,
        "read_fidelity_p25": 0.315,
        "write_fit_r2": 0.0228,
        "write_ridge": 0.01,
        "write_r2_cv": -0.0003,
        "neurons": [
          7654,
          8168,
          3485,
          7299,
          3731,
          8022,
          7206,
          3753,
          2234,
          1308,
          2977,
          2278,
          4660,
          5810,
          1630,
          424
        ]
      }
    },
    "28": {
      "donor_layer": 31,
      "cka": 0.738,
      "read_map_ridge": 1.0,
      "read_map_r2_fit": 0.503,
      "read_map_r2_cv_requests": 0.205,
      "read_map_r2_heldout": 0.22,
      "transplant": {
        "read_fidelity_median": 0.492,
        "read_fidelity_p25": 0.375,
        "write_fit_r2": 0.0022,
        "write_ridge": 10.0,
        "write_r2_cv": -0.0009,
        "neurons": [
          564,
          273,
          1852,
          2527,
          6912,
          109,
          4553,
          2051,
          143,
          7097,
          485,
          192,
          540,
          2532,
          105,
          674
        ]
      },
      "control_random": {
        "read_fidelity_median": 0.404,
        "read_fidelity_p25": 0.346,
        "write_fit_r2": 0.0205,
        "write_ridge": 0.01,
        "write_r2_cv": 0.0026,
        "neurons": [
          4961,
          245,
          4048,
          6627,
          1588,
          5059,
          7961,
          77,
          7092,
          7207,
          6521,
          8012,
          8460,
          675,
          3617,
          3680
        ]
      }
    }
  },
  "nll": {
    "unedited": {
      "fit": 1.07494,
      "test": 1.38426
    },
    "transplant": {
      "scale": 1.0,
      "fit_by_scale": {
        "0.25": 1.07513,
        "0.5": 1.07533,
        "1.0": 1.07496,
        "2.0": 1.07515
      },
      "test": 1.38456,
      "test_minus_unedited": 0.0003,
      "ci95": [
        1e-05,
        0.0006
      ]
    },
    "control_random": {
      "scale": 0.5,
      "fit_by_scale": {
        "0.25": 1.07523,
        "0.5": 1.07474,
        "1.0": 1.07506,
        "2.0": 1.07518
      },
      "test": 1.38429,
      "test_minus_unedited": 3e-05,
      "ci95": [
        -0.00028,
        0.00035
      ]
    },
    "control_clear": {
      "scale": 1.0,
      "fit_by_scale": {
        "1.0": 1.07517
      },
      "test": 1.38447,
      "test_minus_unedited": 0.00021,
      "ci95": [
        -0.00014,
        0.00054
      ]
    },
    "transplant_minus_control_random": {
      "mean": 0.00027,
      "ci95": [
        -3e-05,
        0.00058
      ]
    }
  },
  "gate_passed": false,
  "headline": "held-out code NLL: unedited 1.38426, transplant 1.38456 (diff 0.0003 [1e-05, 0.0006]), random-Qwen control 1.38429, cleared slots 1.38447; gate not passed",
  "system_monitor": {
    "samples": 8
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

## Reading (2026-11-08 07:05 UTC)

Run 3 (Colab L4, 7 minutes of compute; runs 1 and 2 failed on GPU memory before any loss result, smoke runs before that).
360 fit chunks (fastapi, requests), 120 held-out chunks (rich); 8,640 and 2,880 aligned positions.
- Layer match: E2B 25 with Qwen 30 (CKA 0.741) and E2B 28 with Qwen 31 (CKA 0.738).
- Read side: ridge strength 1 chosen by cross-validation inside the fit repositories; read map R^2 0.49 to 0.50 on fit positions,
  0.20 to 0.21 on requests (cross-validation), 0.22 on rich (held out). Read fidelity, median correlation between a transplanted
  neuron in E2B and the Qwen neuron itself on rich: 0.45 and 0.49 for the selected neurons, 0.40 for random ones.
- Write side: the installed neurons' activations predict E2B's correct-next-token direction with R^2 0.017 and 0.002 on fit
  positions and -0.0004 and -0.0009 under cross-validation: no predictive signal.
- Held-out next-token loss on rich (mean over 120 chunks): unedited 1.38426; transplant 1.38456 (+0.0003, 95% interval +0.00001 to
  +0.0006); random Qwen neurons 1.38429 (+0.00003, -0.0003 to +0.0004); slots cleared 1.38447 (+0.0002, -0.0001 to +0.0005).
Gate not passed: the transplant is marginally worse than unedited. No MBPP or agent runs follow (pre-registered).
Reading: a transplanted neuron computes a new nonlinear function of E2B's own state (its read is mapped from E2B's activations),
and those functions carry no linear information about what E2B gets wrong (write R^2 about 0 under cross-validation). The same
limit appeared from the other side in EXP-082: what 31B would do is not readable from E2B's activations. Closed-form neuron
transfer without training can carry readable behaviour (EXP-080) but did not carry capability, within family (EXP-077) or across
families (here).

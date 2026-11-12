# EXP-108: transplant-scaled-18-layers

- **Status:** OK
- **Category:** improvement
- **Started / finished (UTC):** 2026-11-09T16:16:01.812070Z / 2026-11-09T16:59:16.246044Z
- **Wall time:** 43.2 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-102, EXP-106
- **Command:** `experiments/scripts/exp108_transplant_scaled.py`

## Why this experiment
EXP-106's 7-layer E4B transplant moved reference-fix tokens of unseen tasks by a point beyond its control; more layers may move them more.

## Hypothesis (stated before running)
An 18-layer transplant raises held reference-edit token accuracy above the 7-layer one (90.09%) and above its random control.

## Configuration
```json
{
  "e2b_layers": [
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34
  ],
  "arms": [
    "st18_a1",
    "st18_a2",
    "st18_rnd_a1"
  ],
  "eval_states": 160,
  "neurons_per_layer": 128
}
```

## Results
```json
{
  "summary": {
    "base": {
      "tokens": 4286,
      "accuracy": 0.8892,
      "mean_logprob": -0.399
    },
    "st18_a1": {
      "tokens": 4286,
      "accuracy": 0.8964,
      "mean_logprob": -0.382,
      "tokens_gained": 111,
      "tokens_lost": 80,
      "p_tokens": 0.0297
    },
    "st18_a2": {
      "tokens": 4286,
      "accuracy": 0.8591,
      "mean_logprob": -0.6014,
      "tokens_gained": 163,
      "tokens_lost": 292,
      "p_tokens": 0.0
    },
    "st18_rnd_a1": {
      "tokens": 4286,
      "accuracy": 0.8906,
      "mean_logprob": -0.4028,
      "tokens_gained": 29,
      "tokens_lost": 23,
      "p_tokens": 0.4885
    },
    "st7_a1": {
      "tokens": 4286,
      "accuracy": 0.9004,
      "mean_logprob": -0.3574,
      "tokens_gained": 75,
      "tokens_lost": 27,
      "p_tokens": 0.0
    },
    "st7_rnd_a1": {
      "tokens": 4286,
      "accuracy": 0.8901,
      "mean_logprob": -0.4015,
      "tokens_gained": 16,
      "tokens_lost": 12,
      "p_tokens": 0.5716
    }
  },
  "layers": {
    "2": {
      "donor_layer": 2,
      "P_r2_heldout": 0.7768,
      "R_r2_heldout": 0.7598,
      "donor_neurons": 10240,
      "corr2_median": 0.6601,
      "corr2_ge_025": 10015,
      "rms_y_e2b": 0.7425,
      "rms_y_donor": 1.3415,
      "replaced_e2b_contrib_share": 0.00675,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0,
      "st_rnd_predicted_margin_gain_a1": 0.0,
      "st_selected_corr2_median": 0.8696,
      "st_write_norm_ratio_selected": 0.9546
    },
    "4": {
      "donor_layer": 5,
      "P_r2_heldout": 0.7423,
      "R_r2_heldout": 0.8049,
      "donor_neurons": 10240,
      "corr2_median": 0.4695,
      "corr2_ge_025": 7822,
      "rms_y_e2b": 0.9747,
      "rms_y_donor": 1.6953,
      "replaced_e2b_contrib_share": 0.00123,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0,
      "st_rnd_predicted_margin_gain_a1": -0.0,
      "st_selected_corr2_median": 0.8189,
      "st_write_norm_ratio_selected": 0.8912
    },
    "6": {
      "donor_layer": 7,
      "P_r2_heldout": 0.7205,
      "R_r2_heldout": 0.7511,
      "donor_neurons": 10240,
      "corr2_median": 0.5741,
      "corr2_ge_025": 10053,
      "rms_y_e2b": 0.2208,
      "rms_y_donor": 0.7314,
      "replaced_e2b_contrib_share": 0.01095,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0,
      "st_rnd_predicted_margin_gain_a1": 0.0,
      "st_selected_corr2_median": 0.7879,
      "st_write_norm_ratio_selected": 0.9325
    },
    "8": {
      "donor_layer": 10,
      "P_r2_heldout": 0.6711,
      "R_r2_heldout": 0.6632,
      "donor_neurons": 10240,
      "corr2_median": 0.5569,
      "corr2_ge_025": 10157,
      "rms_y_e2b": 0.1311,
      "rms_y_donor": 0.1704,
      "replaced_e2b_contrib_share": 0.01009,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0,
      "st_rnd_predicted_margin_gain_a1": -0.0,
      "st_selected_corr2_median": 0.6773,
      "st_write_norm_ratio_selected": 1.038
    },
    "10": {
      "donor_layer": 12,
      "P_r2_heldout": 0.6218,
      "R_r2_heldout": 0.5946,
      "donor_neurons": 10240,
      "corr2_median": 0.4995,
      "corr2_ge_025": 9966,
      "rms_y_e2b": 0.1108,
      "rms_y_donor": 0.2884,
      "replaced_e2b_contrib_share": 0.00986,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0,
      "st_rnd_predicted_margin_gain_a1": -0.0,
      "st_selected_corr2_median": 0.6299,
      "st_write_norm_ratio_selected": 0.904
    },
    "12": {
      "donor_layer": 14,
      "P_r2_heldout": 0.5671,
      "R_r2_heldout": 0.6175,
      "donor_neurons": 10240,
      "corr2_median": 0.4259,
      "corr2_ge_025": 9911,
      "rms_y_e2b": 0.0911,
      "rms_y_donor": 0.109,
      "replaced_e2b_contrib_share": 0.00777,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0,
      "st_rnd_predicted_margin_gain_a1": -0.0,
      "st_selected_corr2_median": 0.5539,
      "st_write_norm_ratio_selected": 0.9495
    },
    "14": {
      "donor_layer": 17,
      "P_r2_heldout": 0.4613,
      "R_r2_heldout": 0.6973,
      "donor_neurons": 10240,
      "corr2_median": 0.3594,
      "corr2_ge_025": 8520,
      "rms_y_e2b": 0.1205,
      "rms_y_donor": 0.0501,
      "replaced_e2b_contrib_share": 0.00677,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0002,
      "st_rnd_predicted_margin_gain_a1": 0.0,
      "st_selected_corr2_median": 0.4682,
      "st_write_norm_ratio_selected": 1.3495
    },
    "16": {
      "donor_layer": 19,
      "P_r2_heldout": 0.4712,
      "R_r2_heldout": 0.6697,
      "donor_neurons": 10240,
      "corr2_median": 0.3515,
      "corr2_ge_025": 8010,
      "rms_y_e2b": 0.091,
      "rms_y_donor": 0.0703,
      "replaced_e2b_contrib_share": 0.00442,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0013,
      "st_rnd_predicted_margin_gain_a1": 0.0,
      "st_selected_corr2_median": 0.5078,
      "st_write_norm_ratio_selected": 1.2665
    },
    "18": {
      "donor_layer": 22,
      "P_r2_heldout": 0.4652,
      "R_r2_heldout": 0.6544,
      "donor_neurons": 10240,
      "corr2_median": 0.3991,
      "corr2_ge_025": 8874,
      "rms_y_e2b": 0.1253,
      "rms_y_donor": 0.0849,
      "replaced_e2b_contrib_share": 0.00443,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0038,
      "st_rnd_predicted_margin_gain_a1": 0.0002,
      "st_selected_corr2_median": 0.5312,
      "st_write_norm_ratio_selected": 1.2815
    },
    "20": {
      "donor_layer": 24,
      "P_r2_heldout": 0.6997,
      "R_r2_heldout": 0.7602,
      "donor_neurons": 10240,
      "corr2_median": 0.6382,
      "corr2_ge_025": 10237,
      "rms_y_e2b": 0.1383,
      "rms_y_donor": 0.1036,
      "replaced_e2b_contrib_share": 0.00417,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0189,
      "st_rnd_predicted_margin_gain_a1": -0.0004,
      "st_selected_corr2_median": 0.7161,
      "st_write_norm_ratio_selected": 0.9975
    },
    "22": {
      "donor_layer": 27,
      "P_r2_heldout": 0.7016,
      "R_r2_heldout": 0.747,
      "donor_neurons": 10240,
      "corr2_median": 0.6019,
      "corr2_ge_025": 10236,
      "rms_y_e2b": 0.1065,
      "rms_y_donor": 0.1159,
      "replaced_e2b_contrib_share": 0.00494,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.0389,
      "st_rnd_predicted_margin_gain_a1": -0.0005,
      "st_selected_corr2_median": 0.72,
      "st_write_norm_ratio_selected": 0.958
    },
    "24": {
      "donor_layer": 29,
      "P_r2_heldout": 0.7118,
      "R_r2_heldout": 0.8191,
      "donor_neurons": 10240,
      "corr2_median": 0.6399,
      "corr2_ge_025": 10236,
      "rms_y_e2b": 0.143,
      "rms_y_donor": 0.099,
      "replaced_e2b_contrib_share": 0.00379,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.4528,
      "st_rnd_predicted_margin_gain_a1": 0.0007,
      "st_selected_corr2_median": 0.7825,
      "st_write_norm_ratio_selected": 0.9997
    },
    "26": {
      "donor_layer": 31,
      "P_r2_heldout": 0.7418,
      "R_r2_heldout": 0.8143,
      "donor_neurons": 10240,
      "corr2_median": 0.6188,
      "corr2_ge_025": 10236,
      "rms_y_e2b": 0.2672,
      "rms_y_donor": 0.12,
      "replaced_e2b_contrib_share": 0.00321,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.2837,
      "st_rnd_predicted_margin_gain_a1": -0.0039,
      "st_selected_corr2_median": 0.6867,
      "st_write_norm_ratio_selected": 0.8416
    },
    "28": {
      "donor_layer": 34,
      "P_r2_heldout": 0.7355,
      "R_r2_heldout": 0.8132,
      "donor_neurons": 10240,
      "corr2_median": 0.6092,
      "corr2_ge_025": 10228,
      "rms_y_e2b": 0.1977,
      "rms_y_donor": 0.2061,
      "replaced_e2b_contrib_share": 0.00532,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.2622,
      "st_rnd_predicted_margin_gain_a1": -0.0016,
      "st_selected_corr2_median": 0.7082,
      "st_write_norm_ratio_selected": 0.8995
    },
    "30": {
      "donor_layer": 36,
      "P_r2_heldout": 0.7345,
      "R_r2_heldout": 0.7951,
      "donor_neurons": 10240,
      "corr2_median": 0.6058,
      "corr2_ge_025": 10237,
      "rms_y_e2b": 0.1224,
      "rms_y_donor": 0.0737,
      "replaced_e2b_contrib_share": 0.00653,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.4939,
      "st_rnd_predicted_margin_gain_a1": -0.0005,
      "st_selected_corr2_median": 0.691,
      "st_write_norm_ratio_selected": 0.8482
    },
    "32": {
      "donor_layer": 39,
      "P_r2_heldout": 0.7114,
      "R_r2_heldout": 0.771,
      "donor_neurons": 10240,
      "corr2_median": 0.5753,
      "corr2_ge_025": 10235,
      "rms_y_e2b": 0.0681,
      "rms_y_donor": 0.0675,
      "replaced_e2b_contrib_share": 0.00657,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 0.3518,
      "st_rnd_predicted_margin_gain_a1": 0.0089,
      "st_selected_corr2_median": 0.6465,
      "st_write_norm_ratio_selected": 0.8941
    },
    "34": {
      "donor_layer": 41,
      "P_r2_heldout": 0.7034,
      "R_r2_heldout": 0.8158,
      "donor_neurons": 10240,
      "corr2_median": 0.58,
      "corr2_ge_025": 10152,
      "rms_y_e2b": 0.2582,
      "rms_y_donor": 0.1767,
      "replaced_e2b_contrib_share": 0.00333,
      "st_selected": 128,
      "st_predicted_margin_gain_a1": 1.2542,
      "st_rnd_predicted_margin_gain_a1": -0.0206,
      "st_selected_corr2_median": 0.757,
      "st_write_norm_ratio_selected": 1.0893
    }
  },
  "headline": "18-layer E4B transplant, held reference-edit token accuracy: base 0.8892, st18_a1 0.8964, st18_a2 0.8591, st18_rnd_a1 0.8906, st7_a1 0.9004, st7_rnd_a1 0.8901",
  "system_monitor": {
    "samples": 44
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

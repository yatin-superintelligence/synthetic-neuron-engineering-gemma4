# EXP-077: agent-transplant-e4b-to-e2b

- **Status:** OK
- **Category:** transplant
- **Started / finished (UTC):** 2026-11-07T21:59:53.996206Z / 2026-11-07T22:07:57.979922Z
- **Wall time:** 8.1 min
- **Commit:** `` (dirty=False)
- **Depends on:** EXP-063, EXP-071
- **Command:** `experiments/scripts/exp077_agent_transplant_e4b.py`

## Why this experiment
E4B handles the agent states that trap E2B. Transplant E4B neurons tuned to the tool-call segment into E2B's least-used slots, mapped on real agent tokens.

## Hypothesis (stated before running)
Ridge maps reach held-out R^2 above 0.5 at matched depths; selected donor neurons read with median fidelity above 0.5, far above shuffled tokens. Real-task effects are measured in the follow-up runs.

## Configuration
```json
{
  "rec_layers": [
    14,
    15,
    16,
    17,
    18,
    19
  ],
  "top_n": 64,
  "fidelity_min": 0.3,
  "ctx": 1500,
  "max_tokens": 160,
  "runs": [
    "exp029",
    "exp037",
    "exp-044a"
  ]
}
```

## Results
```json
{
  "alignment": {
    "14": {
      "donor_layer": 23,
      "cka": 0.9231,
      "r2_x_fit": 0.7522566318511963,
      "r2_x_hold": 0.5256,
      "r2_r_fit": 0.8363272249698639,
      "r2_r_hold": 0.6658,
      "donor_d_median": 1.153,
      "fid_top_median": 0.8536,
      "fid_rand_median": 0.7507,
      "fid_shuffled_median": 0.0021,
      "n_top_fid_above_min": 64,
      "slot_usage_max": 0.01274,
      "usage_median": 0.02705
    },
    "15": {
      "donor_layer": 24,
      "cka": 0.9103,
      "r2_x_fit": 0.7707345336675644,
      "r2_x_hold": 0.5374,
      "r2_r_fit": 0.8651915937662125,
      "r2_r_hold": 0.6944,
      "donor_d_median": 1.006,
      "fid_top_median": 0.8196,
      "fid_rand_median": 0.7177,
      "fid_shuffled_median": 0.0016,
      "n_top_fid_above_min": 64,
      "slot_usage_max": 0.01873,
      "usage_median": 0.04078
    },
    "16": {
      "donor_layer": 25,
      "cka": 0.8936,
      "r2_x_fit": 0.7683734595775604,
      "r2_x_hold": 0.5264,
      "r2_r_fit": 0.8471129238605499,
      "r2_r_hold": 0.6328,
      "donor_d_median": 0.821,
      "fid_top_median": 0.7471,
      "fid_rand_median": 0.6726,
      "fid_shuffled_median": 0.001,
      "n_top_fid_above_min": 64,
      "slot_usage_max": 0.01194,
      "usage_median": 0.0231
    },
    "17": {
      "donor_layer": 26,
      "cka": 0.8351,
      "r2_x_fit": 0.7537667602300644,
      "r2_x_hold": 0.5023,
      "r2_r_fit": 0.8338718712329865,
      "r2_r_hold": 0.6105,
      "donor_d_median": 0.842,
      "fid_top_median": 0.7513,
      "fid_rand_median": 0.619,
      "fid_shuffled_median": 0.0014,
      "n_top_fid_above_min": 64,
      "slot_usage_max": 0.01642,
      "usage_median": 0.03033
    },
    "18": {
      "donor_layer": 27,
      "cka": 0.8119,
      "r2_x_fit": 0.7296582162380219,
      "r2_x_hold": 0.4791,
      "r2_r_fit": 0.8064771741628647,
      "r2_r_hold": 0.5758,
      "donor_d_median": 0.788,
      "fid_top_median": 0.7393,
      "fid_rand_median": 0.6124,
      "fid_shuffled_median": 0.0035,
      "n_top_fid_above_min": 64,
      "slot_usage_max": 0.01275,
      "usage_median": 0.02417
    },
    "19": {
      "donor_layer": 28,
      "cka": 0.8339,
      "r2_x_fit": 0.7191120982170105,
      "r2_x_hold": 0.4865,
      "r2_r_fit": 0.7832572758197784,
      "r2_r_hold": 0.5789,
      "donor_d_median": 0.779,
      "fid_top_median": 0.7399,
      "fid_rand_median": 0.6384,
      "fid_shuffled_median": -0.0015,
      "n_top_fid_above_min": 64,
      "slot_usage_max": 0.01241,
      "usage_median": 0.02607
    }
  },
  "adapters": {
    "tx_call64_k1.0": {
      "14": 64,
      "15": 64,
      "16": 64,
      "17": 64,
      "18": 64,
      "19": 64
    },
    "tx_call64_k0.5": {
      "14": 64,
      "15": 64,
      "16": 64,
      "17": 64,
      "18": 64,
      "19": 64
    },
    "tx_rand64_k1.0": {
      "14": 64,
      "15": 64,
      "16": 64,
      "17": 64,
      "18": 64,
      "19": 64
    },
    "recycle_only64": {
      "14": 64,
      "15": 64,
      "16": 64,
      "17": 64,
      "18": 64,
      "19": 64
    }
  },
  "headline": "median held-out R2(x) 0.514; read fidelity of selected donor neurons 0.749, random 0.655, shuffled 0.002; installed per layer {14: 64, 15: 64, 16: 64, 17: 64, 18: 64, 19: 64}",
  "system_monitor": {
    "samples": 9
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

## Real-task test, pre-registered 2026-11-07 22:15 UTC, before any run

Adapters checked against the base weights: each installed neuron matches its mapped donor neuron to within 0.0013 (relative, float16
storage), and no weight outside the 64 chosen slots per layer changes. Arms on the official stack (E2B W4A16, 58 small-fix tasks,
official settings), seeds 1, 2 and 3, on a Colab A100:
- `tx_call64_k1.0`: the transplant.
- `tx_rand64_k1.0`: 64 random donor neurons through the same maps and slots (does any donor content help?).
- `recycle_only64`: the same slots emptied, nothing installed (what freeing the slots costs).
- No edit: EXP-044a, EXP-044b, EXP-044c (seeds 1 to 3).
- `tx_call64_k0.5` runs only if the transplant at full strength shows a gain or clear harm.

Measures: primary, tasks solved summed over the 3 seeds, against no edit, the random control and the recycle control. Secondary: edit
calls applied, context errors, loops, tool calls. A gain counts only if it beats all three comparisons and holds by task across seeds.

## First real-task reading (2026-11-08 00:45 UTC)

EXP-077a, tx_call64_k1.0 (64 E4B agent neurons installed into E2B's least-used slots, layers 14 to 19, full strength), official E2B
W4A16, seed 1, Colab A100. The transplant breaks generation: 49 of 58 tasks had a thinking loop (171 loops), against 21 of 58 in
each unedited run (EXP-044b 34 loops, EXP-044e 38); median output 16,744 tokens per task against 5,442 and 7,285; 36 of 58 tasks
ended in a context overflow (unedited 12 to 18); 19 of 114 edits applied (unedited about half). Official score pending on Kaggle.
The effect is far beyond seed noise, so seeds 2 and 3 of the same arms were dropped before running. Kept at seed 1, to find which
part does the damage: EXP-077b (64 random E4B neurons, same install), EXP-077c (the recipient slots cleared, nothing installed) and
EXP-077j (the same 64 neurons at half strength).

EXP-077b (64 random E4B neurons, same layers, slots and strength, seed 1): normal. 12 of 58 context overflows, 53 of 128 edits
applied, against 36 and 19 of 114 for the selected neurons. Installing 64 neurons does not by itself break E2B; what the selected
agent neurons write does.

EXP-077c (the 64 recipient slots cleared, nothing installed, seed 1): normal. 15 of 58 context overflows, 51 of 117 edits applied.
Freeing E2B's least-used neurons is harmless; with EXP-077b (random donor neurons) this places the damage in the selected neurons'
writes. EXP-077j (the same 64 at half strength) tests whether a weaker write keeps any benefit without the loops.

EXP-077j (the 64 selected neurons at half strength, seed 1): 31 of 58 tasks with a loop (61 loops), 23 context overflows, 36 of 87
edits applied. The damage scales with the write strength: full strength 49 of 58 tasks looping, half strength 31, random donor
neurons 22, slots cleared only 21 (unedited runs 18 to 28). Conclusion for the paper: a transplant of selected E4B neurons into E2B
is read faithfully (read fidelity 0.74 to 0.85) but its mapped writes push E2B into thinking loops in proportion to their strength,
and no strength tested helps. The E4B-to-E2B transplant list is finished; no further transplant runs are planned.

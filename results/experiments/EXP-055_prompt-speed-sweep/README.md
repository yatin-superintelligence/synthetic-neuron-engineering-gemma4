# EXP-055: prompt-speed-sweep

- **Status:** OK
- **Category:** infrastructure
- **Started / finished (UTC):** 2026-11-06T09:13:59.831733Z / 2026-11-06T09:27:11.414953Z
- **Wall time:** 13.2 min
- **Commit:** `8a4cdf9b1df2ffc85d55b009cad653e6e3a57994` (dirty=True)
- **Depends on:** EXP-001
- **Command:** `experiments/scripts/exp055_prompt_speed_sweep.py`

## Why this experiment
I accepted 130 tokens/s on long prompts without testing it, then cut the states I test to fit. If the rate can be raised, local replays need no such cut and the need for paid compute falls.

## Hypothesis (stated before running)
A larger physical batch raises prompt throughput, because the GPU is underused at 512 tokens per step.

## Configuration
```json
{
  "grid": [
    {
      "ub": 512,
      "b": 2048,
      "np": 6
    },
    {
      "ub": 1024,
      "b": 2048,
      "np": 6
    },
    {
      "ub": 2048,
      "b": 2048,
      "np": 6
    },
    {
      "ub": 4096,
      "b": 4096,
      "np": 6
    },
    {
      "ub": 8192,
      "b": 8192,
      "np": 6
    },
    {
      "ub": 512,
      "b": 2048,
      "np": 1
    },
    {
      "ub": 2048,
      "b": 2048,
      "np": 1
    },
    {
      "ub": 512,
      "b": 2048,
      "np": 6
    }
  ],
  "target_prompt_tokens": 10000
}
```

## Results
```json
{
  "rows": [
    {
      "ub": 512,
      "b": 2048,
      "np": 6,
      "precision": "q4_0",
      "prompt_tps": 267.5,
      "gen_tps": 14.2,
      "wall_s": 40.0,
      "n_prompt": 10094,
      "package_w": 9.5,
      "igpu_w": 0,
      "freq_mhz": 1695,
      "status": "ok"
    },
    {
      "ub": 1024,
      "b": 2048,
      "np": 6,
      "precision": "q4_0",
      "prompt_tps": 281.2,
      "gen_tps": 14.75,
      "wall_s": 38.0,
      "n_prompt": 10094,
      "package_w": 9.1,
      "igpu_w": 0,
      "freq_mhz": 1699,
      "status": "ok"
    },
    {
      "ub": 2048,
      "b": 2048,
      "np": 6,
      "precision": "q4_0",
      "prompt_tps": 280.7,
      "gen_tps": 14.96,
      "wall_s": 38.1,
      "n_prompt": 10094,
      "package_w": 10.6,
      "igpu_w": 0,
      "freq_mhz": 1601,
      "status": "ok"
    },
    {
      "ub": 4096,
      "b": 4096,
      "np": 6,
      "precision": "q4_0",
      "status": "<HTTPError 500: 'Internal Server Error'>"
    },
    {
      "ub": 8192,
      "b": 8192,
      "np": 6,
      "precision": "q4_0",
      "status": "<HTTPError 500: 'Internal Server Error'>"
    },
    {
      "ub": 512,
      "b": 2048,
      "np": 1,
      "precision": "q4_0",
      "prompt_tps": 651.5,
      "gen_tps": 18.97,
      "wall_s": 17.2,
      "n_prompt": 10094,
      "package_w": 11.1,
      "igpu_w": 0,
      "freq_mhz": 1699,
      "status": "ok"
    },
    {
      "ub": 2048,
      "b": 2048,
      "np": 1,
      "precision": "q4_0",
      "prompt_tps": 660.4,
      "gen_tps": 15.57,
      "wall_s": 17.3,
      "n_prompt": 10094,
      "package_w": 6.1,
      "igpu_w": 0,
      "freq_mhz": 1699,
      "status": "ok"
    },
    {
      "ub": 512,
      "b": 2048,
      "np": 6,
      "precision": "q4_0",
      "prompt_tps": 235.9,
      "gen_tps": 14.75,
      "wall_s": 44.9,
      "n_prompt": 10094,
      "package_w": 9.9,
      "igpu_w": 0,
      "freq_mhz": 1820,
      "status": "ok"
    },
    {
      "ub": 512,
      "b": 2048,
      "np": 6,
      "precision": "f16",
      "prompt_tps": 165.0,
      "gen_tps": 10.11,
      "wall_s": 64.3,
      "n_prompt": 10094,
      "package_w": 12.3,
      "igpu_w": 0.1,
      "freq_mhz": 1913,
      "status": "ok"
    },
    {
      "ub": 1024,
      "b": 2048,
      "np": 6,
      "precision": "f16",
      "prompt_tps": 182.1,
      "gen_tps": 10.06,
      "wall_s": 58.6,
      "n_prompt": 10094,
      "package_w": 19.2,
      "igpu_w": 0,
      "freq_mhz": 1769,
      "status": "ok"
    },
    {
      "ub": 2048,
      "b": 2048,
      "np": 6,
      "precision": "f16",
      "status": "ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None)"
    },
    {
      "ub": 4096,
      "b": 4096,
      "np": 6,
      "precision": "f16",
      "status": "ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None)"
    },
    {
      "ub": 8192,
      "b": 8192,
      "np": 6,
      "precision": "f16",
      "status": "ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None)"
    },
    {
      "ub": 512,
      "b": 2048,
      "np": 1,
      "precision": "f16",
      "prompt_tps": 524.9,
      "gen_tps": 10.62,
      "wall_s": 22.2,
      "n_prompt": 10094,
      "package_w": 7.1,
      "igpu_w": 0,
      "freq_mhz": 1768,
      "status": "ok"
    },
    {
      "ub": 2048,
      "b": 2048,
      "np": 1,
      "precision": "f16",
      "prompt_tps": 515.8,
      "gen_tps": 10.56,
      "wall_s": 22.5,
      "n_prompt": 10094,
      "package_w": 7.3,
      "igpu_w": 0,
      "freq_mhz": 1764,
      "status": "ok"
    },
    {
      "ub": 512,
      "b": 2048,
      "np": 6,
      "precision": "f16",
      "prompt_tps": 196.7,
      "gen_tps": 10.49,
      "wall_s": 54.3,
      "n_prompt": 10094,
      "package_w": 4.5,
      "igpu_w": 0,
      "freq_mhz": 1876,
      "status": "ok"
    }
  ],
  "headline": "best prompt tokens/s: q4_0 660.4 at ub 2048, f16 524.9 at ub 512; reference ub 512: [267.5, 235.9, 165.0, 196.7]",
  "system_monitor": {
    "samples": 12,
    "ram_free_gb": {
      "min": 5.04,
      "max": 19.91,
      "mean": 12.32
    },
    "gpu_shared_gb": {
      "min": 1.34,
      "max": 12.01,
      "mean": 6.58
    },
    "gpu_util_pct": {
      "min": 0,
      "max": 100,
      "mean": 56.67
    },
    "cpu_package_w": {
      "min": 17.9,
      "max": 40.2,
      "mean": 27.59
    },
    "cpu_core_w": {
      "min": 1,
      "max": 24.7,
      "mean": 9.43
    },
    "igpu_w": {
      "min": 0,
      "max": 21.3,
      "mean": 11.56
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0,
      "max": 17.5,
      "mean": 10.31
    },
    "cpu_perf_pct": {
      "min": 52.8,
      "max": 150.5,
      "mean": 92.7
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
      "min": 1662,
      "max": 1876,
      "mean": 1784.0
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
- **Reserved slots are the main cost.** With one request running, prompts go at 652 tokens/s with 1 slot and 236 to 268 with 6
  on Q4_0, and 525 against 165 to 197 on F16. Reserving slots costs 2.4 to 2.7 times, even when they sit idle. The earlier
  130 tokens/s came from this setting plus active slots competing, not from the hardware. Generation gains too: Q4_0 19 against 14
  tokens/s; F16 is flat at about 10.5.
- **Batch size barely matters.** 1024 adds about 5% at 6 slots, and nothing at 1 slot.
- **Large batches crash the GPU.** Q4_0 at 4096 and 8192 fail with Vulkan ErrorDeviceLost, and at F16 the 2048, 4096 and 8192 runs
  at 6 slots failed too. Those failed runs are recorded as they happened. The safe range here is at most 1024 at 6 slots.
- **Drift check.** The repeated reference gave 267.5 then 235.9 on Q4_0 and 165.0 then 196.7 on F16, so differences below about
  20% between single measurements here are noise. The 2.4 times slot effect is well above that; the 5% batch effect is not.
- One prompt was measured per setting. EXP-057 measures whole batches, which is the job that matters.
- The recorded package power was read after each request ended, so it is idle power and says nothing about load.

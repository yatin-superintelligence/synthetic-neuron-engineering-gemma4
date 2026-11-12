# EXP-007: openvino-engine-comparison

- **Status:** OK
- **Category:** infrastructure
- **Started / finished (UTC):** 2026-11-05T21:53:21.153157Z / 2026-11-05T22:04:13.431669Z
- **Wall time:** 10.9 min
- **Commit:** `c2355a66ec4b19fa60bc46335082203f506953d4` (dirty=True)
- **Depends on:** EXP-001
- **Command:** `experiments/scripts/exp007_openvino_engines.py`

## Why this experiment
The author asked for the fastest engine on the Arc GPU; EXP-001 did not test OpenVINO.

## Hypothesis (stated before running)
OpenVINO GenAI is faster single-stream at int4; the llama.cpp OpenVINO backend is at best on par with Vulkan at 8 parallel.

## Configuration
```json
{
  "ov_model": "OpenVINO/gemma-4-E2B-it-int4-ov",
  "llama": "b12129",
  "npp": 256,
  "ntg": 128,
  "npl": [
    1,
    8
  ]
}
```

## Results
```json
{
  "vulkan_q4_0": {
    "res": {
      "1": {
        "pp": 563.41,
        "tg": 26.15
      },
      "8": {
        "pp": 615.68,
        "tg": 120.45
      }
    },
    "wall": 21.0,
    "rc": 0
  },
  "llama_openvino_q4_0": {
    "res": {
      "1": {
        "pp": 227.56,
        "tg": 6.62
      },
      "8": {
        "pp": 119.25,
        "tg": 72.95
      }
    },
    "wall": 81.0,
    "rc": 0
  },
  "vulkan_f16": {
    "res": {
      "1": {
        "pp": 298.19,
        "tg": 14.28
      },
      "8": {
        "pp": 333.68,
        "tg": 80.21
      }
    },
    "wall": 37.2,
    "rc": 0
  },
  "llama_openvino_f16": {
    "res": {
      "1": {
        "pp": 58.58,
        "tg": 2.64
      },
      "8": {
        "pp": 145.56,
        "tg": 27.59
      }
    },
    "wall": 137.6,
    "rc": 0
  },
  "openvino_genai_int4": {
    "single_tps_wall": 35.43,
    "tps_metric": 35.97,
    "ttft_ms": 81.9
  },
  "headline": "vulkan_q4_0: 120.45; llama_openvino_q4_0: 72.95; vulkan_f16: 80.21; llama_openvino_f16: 27.59; openvino_genai_int4: {'single_tps_wall': 35.43, 'tps_metric': 35.97, 'ttft_ms': 81.9}",
  "system_monitor": {
    "samples": 10,
    "ram_free_gb": {
      "min": 1.24,
      "max": 23.72,
      "mean": 17.57
    },
    "gpu_shared_gb": {
      "min": 1.22,
      "max": 6,
      "mean": 2.04
    },
    "gpu_util_pct": {
      "min": 0.2,
      "max": 96.5,
      "mean": 9.92
    },
    "thermal_c": {
      "min": 27.9,
      "max": 27.9,
      "mean": 27.9
    },
    "battery_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_pct": {
      "min": 0.9,
      "max": 26.9,
      "mean": 7.18
    },
    "cpu_perf_pct": {
      "min": 63.5,
      "max": 170,
      "mean": 105.74
    },
    "cpu_perf_limit_pct": {
      "min": 100,
      "max": 100,
      "mean": 100.0
    },
    "cpu_freq_mhz": {
      "min": 1619,
      "max": 1913,
      "mean": 1803.4
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

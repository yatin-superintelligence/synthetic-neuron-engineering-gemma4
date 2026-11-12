# EXP-001: inference-engine-benchmark

- **Status:** OK
- **Category:** infrastructure
- **Started / finished (UTC):** 2026-11-05T19:23:09.087504Z / 2026-11-05T19:47:12.353180Z
- **Wall time:** 24.1 min
- **Commit:** `98aa1c0f2d00d4631ec15ef9fb74d49722f9c74f` (dirty=False)
- **Depends on:** -
- **Command:** `experiments/scripts/exp001_engine_benchmark.py`

## Why this experiment
Hundreds of generation-based evaluations are planned; throughput on the Arc iGPU decides how many experiments and seeds are affordable. The engine must also support per-request LoRA adapters so edits can be evaluated without reloading the base model.

## Hypothesis (stated before running)
llama.cpp (Vulkan or SYCL) is several times faster than PyTorch XPU eager generation; q4_0 decodes faster than F16 because decode is memory-bandwidth bound.

## Configuration
```json
{
  "models": {
    "q4_0": "<repo>\\models\\gguf\\gemma-4-E2B_q4_0-it.gguf",
    "f16": "<repo>\\models\\gguf\\e2b-qat-f16.gguf"
  },
  "llama_cpp": "b12129",
  "ngl": 99,
  "pp": 256,
  "tg": 256,
  "npl": [
    1,
    4,
    8,
    16,
    32
  ],
  "flash_attn": "on"
}
```

## Results
```json
{
  "llamacpp_vulkan_q4_0": {
    "single": {
      "pp512": 2024.71,
      "tg128": 31.07
    },
    "batched": {
      "1": {
        "pp_tps": 639.7,
        "tg_tps": 30.39,
        "total_tps": 58.03
      },
      "4": {
        "pp_tps": 673.39,
        "tg_tps": 97.81,
        "total_tps": 170.81
      },
      "8": {
        "pp_tps": 671.01,
        "tg_tps": 144.89,
        "total_tps": 238.32
      },
      "16": {
        "pp_tps": 676.13,
        "tg_tps": 95.56,
        "total_tps": 167.45
      },
      "32": {
        "pp_tps": 674.41,
        "tg_tps": 162.27,
        "total_tps": 261.59
      }
    }
  },
  "llamacpp_vulkan_f16": {
    "single": {
      "pp512": 938.35,
      "tg128": 15.87
    },
    "batched": {
      "1": {
        "pp_tps": 302.52,
        "tg_tps": 15.58,
        "total_tps": 29.63
      },
      "4": {
        "pp_tps": 321.38,
        "tg_tps": 57.47,
        "total_tps": 97.51
      },
      "8": {
        "pp_tps": 324.63,
        "tg_tps": 85.8,
        "total_tps": 135.72
      },
      "16": {
        "pp_tps": 329.53,
        "tg_tps": 50.79,
        "total_tps": 88.01
      },
      "32": {
        "pp_tps": 328.22,
        "tg_tps": 88.01,
        "total_tps": 138.8
      }
    }
  },
  "llamacpp_sycl_q4_0": {
    "single": {
      "pp512": 1770.55,
      "tg128": 29.77
    },
    "batched": {
      "1": {
        "pp_tps": 82.15,
        "tg_tps": 28.26,
        "total_tps": 42.05
      },
      "4": {
        "pp_tps": 534.12,
        "tg_tps": 83.55,
        "total_tps": 144.49
      },
      "8": {
        "pp_tps": 541.85,
        "tg_tps": 124.72,
        "total_tps": 202.77
      },
      "16": {
        "pp_tps": 526.68,
        "tg_tps": 49.15,
        "total_tps": 89.91
      },
      "32": {
        "pp_tps": 548.04,
        "tg_tps": 90.84,
        "total_tps": 155.84
      }
    }
  },
  "llamacpp_sycl_f16": {
    "single": {
      "pp512": 1402.3,
      "tg128": 12.55
    },
    "batched": {
      "1": {
        "pp_tps": 148.59,
        "tg_tps": 13.12,
        "total_tps": 24.11
      },
      "4": {
        "pp_tps": 447.32,
        "tg_tps": 8.84,
        "total_tps": 17.34
      },
      "8": {
        "pp_tps": 458.95,
        "tg_tps": 17.07,
        "total_tps": 32.91
      },
      "16": {
        "pp_tps": 462.03,
        "tg_tps": 31.75,
        "total_tps": 59.41
      },
      "32": {
        "pp_tps": 412.96,
        "tg_tps": 57.54,
        "total_tps": 101.0
      }
    }
  },
  "torch_xpu_bf16_eager": {
    "decode_tps_by_batch": {
      "1": 3.17,
      "8": 24.15
    },
    "error": null,
    "wall_s": 63.0
  },
  "headline": "best aggregate decode @16 seqs: llamacpp_vulkan_q4_0 95.56 tok/s; torch eager bs1/bs8: {1: 3.17, 8: 24.15}",
  "system_monitor": {
    "samples": 22,
    "ram_free_gb": {
      "min": 2.2,
      "max": 16.4,
      "mean": 9.39
    },
    "gpu_shared_gb": {
      "min": 2.95,
      "max": 10.26,
      "mean": 6.64
    },
    "gpu_util_pct": {
      "min": 5.9,
      "max": 204.4,
      "mean": 97.68
    },
    "thermal_c": {
      "min": 27.9,
      "max": 27.9,
      "mean": 27.9
    },
    "battery_pct": {
      "min": 32,
      "max": 43,
      "mean": 39.27
    },
    "cpu_pct": {
      "min": 2.7,
      "max": 84.1,
      "mean": 19.36
    }
  }
}
```

## Files
- `run.log` full console log
- `config.json` config + environment provenance
- `results.json` metrics
- `sysmon.jsonl` RAM/GPU/battery/thermal samples every 60 s
- other `*.json` raw outputs (generations, per-problem outcomes)

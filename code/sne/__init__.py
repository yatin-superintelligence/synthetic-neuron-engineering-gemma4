"""sne: Synthetic Neuron Engineering - training-free neuron-level edits for Gemma 4, compiled to LoRA."""
import os
import sys

# Tokens can be any Unicode, and the Windows console default cannot print them all.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Must be set before torch initialises Level Zero (Gemma 4 PLE table is a 4.7 GB single tensor).
os.environ.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")

SMOKE = os.environ.get("SNE_SMOKE") == "1"  # preflight: whole experiment on tiny slices, outputs under scratch/preflight
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")

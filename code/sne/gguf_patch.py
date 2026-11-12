"""In-place patching of small GGUF tensors (e.g. Gemma 4's per-layer `blk.N.layer_output_scale`), with guaranteed
restore. Avoids copying 3-9 GB model files for scalar experiments.

Safety: before writing, the original bytes of every patched tensor are saved to <gguf>.patch-backup.json. On entry,
any leftover backup (from a crash) is restored first. On exit the originals are written back and verified.
Only F32 tensors are supported (layer_output_scale and norm weights are F32 in these files).
"""
import json
import os

import numpy as np


def _tensor_offsets(path, names):
    import gguf
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        if t.name in names:
            if t.tensor_type.name != "F32":
                raise ValueError(f"{t.name} is {t.tensor_type.name}, only F32 supported")
            out[t.name] = (int(t.data_offset), int(t.n_elements))
    missing = set(names) - set(out)
    if missing:
        raise KeyError(f"tensors not found: {sorted(missing)[:5]}")
    return out


def restore_if_needed(path):
    bk = path + ".patch-backup.json"
    if not os.path.exists(bk):
        return False
    rec = json.load(open(bk, encoding="utf-8"))
    with open(path, "r+b") as f:
        for name, (off, vals) in rec.items():
            f.seek(off)
            f.write(np.asarray(vals, dtype=np.float32).tobytes())
    os.remove(bk)
    return True


class Patched:
    """with Patched(gguf_path, {"blk.3.layer_output_scale": np.array([0.7])}): ... run server ..."""

    def __init__(self, path, new_values):
        self.path, self.new = path, {k: np.asarray(v, dtype=np.float32).ravel() for k, v in new_values.items()}

    def __enter__(self):
        restore_if_needed(self.path)
        offs = _tensor_offsets(self.path, set(self.new))
        backup = {}
        with open(self.path, "r+b") as f:
            for name, (off, n) in offs.items():
                f.seek(off)
                old = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
                backup[name] = (off, old.tolist())
        with open(self.path + ".patch-backup.json", "w", encoding="utf-8") as f:
            json.dump(backup, f)
        with open(self.path, "r+b") as f:
            for name, (off, n) in offs.items():
                assert self.new[name].size == n, name
                f.seek(off)
                f.write(self.new[name].tobytes())
        self.backup = backup
        return self

    def __exit__(self, *a):
        restore_if_needed(self.path)
        with open(self.path, "rb") as f:  # verify restore
            for name, (off, old) in self.backup.items():
                f.seek(off)
                now = np.frombuffer(f.read(4 * len(old)), dtype=np.float32)
                assert np.array_equal(now, np.asarray(old, dtype=np.float32)), f"restore failed for {name}"
        return False


def read_values(path, names):
    offs = _tensor_offsets(path, set(names))
    out = {}
    with open(path, "rb") as f:
        for name, (off, n) in offs.items():
            f.seek(off)
            out[name] = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
    return out

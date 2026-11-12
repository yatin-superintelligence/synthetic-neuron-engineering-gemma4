"""Paired statistics for per-problem pass/fail outcomes."""
import math

import numpy as np
from scipy import stats as sps


def paired_compare(base, treat, n_boot=10000, seed=0):
    """base/treat: dict task_id -> bool (same keys). Returns delta, 95% bootstrap CI, exact McNemar p."""
    keys = sorted(set(base) & set(treat))
    a = np.array([base[k] for k in keys], dtype=float)
    b = np.array([treat[k] for k in keys], dtype=float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n01 = int(((a == 0) & (b == 1)).sum())  # fixed by treatment
    n10 = int(((a == 1) & (b == 0)).sum())  # broken by treatment
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sps.binom.cdf(min(n01, n10), n, 0.5))
    return dict(n=len(keys), base=a.mean(), treat=b.mean(), delta=d.mean(), ci95=(lo, hi),
                gained=n01, lost=n10, mcnemar_p=p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) * 2 / 2 ** n
    return min(1.0, p)

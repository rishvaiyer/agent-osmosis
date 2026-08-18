"""Small metric helpers."""
from __future__ import annotations
import numpy as np


def jaccard(a, b) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def mean_ci(values, z: float = 1.96):
    """Mean and half-width of a ~95% CI (normal approx)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None
    m = float(v.mean())
    half = float(z * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, half


def interp_curve(seen, acc, grid):
    """Resample a step->accuracy curve onto a common x grid for averaging trials."""
    return np.interp(grid, seen, acc, left=acc[0], right=acc[-1])

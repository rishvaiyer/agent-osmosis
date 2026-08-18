"""Small metric helpers — the little math utilities the experiments lean on."""
from __future__ import annotations
import numpy as np


def jaccard(a, b) -> float:
    """How much do two sets of items overlap? 0.0 = nothing in common, 1.0 = identical.

    Jaccard = (items in both) / (items in either). We use it to ask "do two models
    pick the same influential examples?" — the higher the overlap, the more they agree.
    """
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:                      # two empty sets count as identical
        return 1.0
    return len(sa & sb) / len(sa | sb)         # intersection over union


def mean_ci(values, z: float = 1.96):
    """Average of some numbers, plus a 95% confidence interval (how sure we are).

    Returns (mean, half_width). A result like "110 ± 11" means the true average is
    very likely between 99 and 121. `None` values (e.g. a trial that never hit target)
    are dropped. z=1.96 is the standard multiplier for a 95% interval.
    """
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None
    m = float(v.mean())
    # standard error of the mean × z → the ± half-width of the interval
    half = float(z * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, half


def interp_curve(seen, acc, grid):
    """Resample a (examples-seen → accuracy) curve onto a shared x-axis.

    Different trials hit their eval checkpoints at slightly different points, so before
    we can average their curves we line them all up on the same set of x-values (`grid`)
    by interpolating. Outside the measured range we hold the first/last value flat.
    """
    return np.interp(grid, seen, acc, left=acc[0], right=acc[-1])

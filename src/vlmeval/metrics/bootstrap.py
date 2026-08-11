"""Percentile bootstrap confidence intervals over per-sample rows."""

from __future__ import annotations

import random
from typing import Any, Callable


def bootstrap_ci(
    rows: list[Any],
    stat_fn: Callable[[list[Any]], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 3407,
) -> tuple[float, float]:
    """(lo, hi) percentile bootstrap CI of `stat_fn` under resampling of `rows`.

    Deterministic for a fixed seed. For mean-of-scores metrics pass the mean;
    for CORD, pass a stat_fn that recomputes micro F1 on the resampled rows.
    """
    if not rows:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(rows)
    stats = sorted(stat_fn([rows[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (stats[lo_idx], stats[hi_idx])

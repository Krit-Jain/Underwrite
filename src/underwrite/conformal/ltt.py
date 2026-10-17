"""Learn-Then-Test threshold certification.

We choose an accept threshold from a **fixed, pre-specified grid** by fixed-sequence
testing: walk from the strictest threshold to the most lenient, and at each one test

    H0(tau): the selective risk among items accepted at tau is greater than alpha

with an exact binomial test at level ``delta``. Stop at the first threshold that
fails to reject. Because the thresholds are tested in a fixed order and we stop at
the first failure, the procedure controls the family-wise error rate, so with
probability at least ``1 - delta`` the population selective risk at the certified
threshold is at most ``alpha`` -- provided new items are exchangeable with the
calibration items.

Nothing here is tuned on the data: the grid is decided in advance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import binom

DEFAULT_GRID_SIZE = 501


@dataclass(frozen=True)
class Certification:
    """Outcome of certifying one error budget."""

    alpha: float
    delta: float
    threshold: float | None
    coverage: float
    n_accepted: int
    n_errors: int

    @property
    def certified(self) -> bool:
        return self.threshold is not None

    @property
    def selective_risk(self) -> float:
        return self.n_errors / self.n_accepted if self.n_accepted else float("nan")


def default_grid(size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    """The fixed threshold grid, ascending."""
    return np.linspace(0.0, 1.0, size)


def certify_threshold(
    scores: Sequence[float],
    errors: Sequence[int],
    alpha: float,
    delta: float,
    grid: Sequence[float],
) -> float | None:
    """Return the certified threshold, or ``None`` when nothing can be certified.

    Thresholds that accept fewer than :func:`minimum_accepts_for_zero_errors` items
    are **skipped**, not tested. Such a threshold cannot reject even with zero
    observed errors, so testing it would only truncate the sequence at a stage
    carrying no information -- with a raw score grid, the strictest non-empty
    threshold typically accepts a single item and halts the whole procedure.
    """
    score_array = np.asarray(scores, dtype=float)
    error_array = np.asarray(errors, dtype=float)
    if score_array.shape != error_array.shape:
        raise ValueError("scores and errors must have the same length")

    minimum = minimum_accepts_for_zero_errors(alpha, delta)
    best: float | None = None
    for threshold in sorted(grid, reverse=True):
        accepted = score_array >= threshold
        count = int(accepted.sum())
        if count < max(1, minimum):
            continue
        mistakes = int(error_array[accepted].sum())
        # Exact binomial test: P[Bin(n, alpha) <= k]; small means the risk is below alpha.
        if binom.cdf(mistakes, count, alpha) <= delta:
            best = float(threshold)
        else:
            break
    return best


def certify(
    scores: Sequence[float],
    errors: Sequence[int],
    alpha: float,
    delta: float,
    grid: Sequence[float] | None = None,
) -> Certification:
    """Certify a threshold and report the coverage it achieves on this sample."""
    thresholds = default_grid() if grid is None else grid
    threshold = certify_threshold(scores, errors, alpha, delta, thresholds)

    score_array = np.asarray(scores, dtype=float)
    error_array = np.asarray(errors, dtype=float)
    if threshold is None:
        return Certification(alpha, delta, None, 0.0, 0, 0)

    accepted = score_array >= threshold
    return Certification(
        alpha=alpha,
        delta=delta,
        threshold=threshold,
        coverage=float(accepted.mean()),
        n_accepted=int(accepted.sum()),
        n_errors=int(error_array[accepted].sum()),
    )


def selective_risk(scores: Sequence[float], errors: Sequence[int], threshold: float) -> float:
    """Error rate among items accepted at ``threshold``."""
    accepted = np.asarray(scores, dtype=float) >= threshold
    if not accepted.any():
        return float("nan")
    return float(np.asarray(errors, dtype=float)[accepted].mean())


def minimum_accepts_for_zero_errors(alpha: float, delta: float) -> int:
    """Smallest number of error-free accepts that certifies ``alpha`` at ``delta``.

    Solves ``(1 - alpha) ** n <= delta``. At alpha=5%, delta=0.1 this is 45 items;
    at 2% it is 114, and at 1% it is 230 -- the sample-size reality of the guarantee.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must lie in (0, 1)")
    return int(np.ceil(np.log(delta) / np.log(1.0 - alpha)))

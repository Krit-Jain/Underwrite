"""Resplit validity of the certified threshold.

The scorers are frozen (fitted on probe_train), then the calibration and test
documents are pooled and repeatedly split at random. For each split the threshold is
certified on one part and its behaviour measured on the other. This asks whether the
certificate is reliable across resamples rather than on one fortunate split.

A caution about interpreting the exceedance rate: the certified claim is about the
*population* selective risk. The realised risk on a finite test part fluctuates
around alpha, so it crosses alpha more often than ``delta`` even when the procedure
is sound. The full distribution of realised risk is reported for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from underwrite.conformal.ltt import certify

OUT_OF_SAMPLE_SPLITS = ("calibration", "test")


@dataclass(frozen=True)
class ValiditySummary:
    """What happens to the certificate across many random resplits."""

    alpha: float
    delta: float
    n_splits: int
    certified_rate: float
    violation_rate: float
    mean_coverage: float
    sd_coverage: float
    mean_realised_risk: float
    sd_realised_risk: float
    p95_realised_risk: float
    n_certified: int = 0

    @property
    def uncertified_rate(self) -> float:
        return 1.0 - self.certified_rate

    @property
    def conditional_violation_rate(self) -> float:
        """Share of *certified* splits whose realised risk exceeded alpha.

        ``violation_rate`` divides by every split, so it is diluted by the splits where
        no certificate was issued at all. This is the rate a user actually cares about:
        given that the procedure made a promise, how often did it break it?
        """
        if self.n_certified == 0:
            return float("nan")
        return self.violation_rate / self.certified_rate


def resplit_validity(
    scores: Sequence[float],
    errors: Sequence[int],
    alpha: float,
    delta: float,
    *,
    n_splits: int = 200,
    cal_fraction: float = 0.5,
    seed: int = 0,
    grid: Sequence[float] | None = None,
) -> ValiditySummary:
    """Certify on random halves of the pool and measure the other half, ``n_splits`` times."""
    score_array = np.asarray(scores, dtype=float)
    error_array = np.asarray(errors, dtype=float)
    if score_array.shape != error_array.shape:
        raise ValueError("scores and errors must have the same length")
    if score_array.size < 4:
        raise ValueError("need at least 4 items to split")
    if not 0.0 < cal_fraction < 1.0:
        raise ValueError("cal_fraction must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    count = score_array.size
    n_cal = max(1, min(count - 1, int(round(count * cal_fraction))))

    certified = 0
    violations = 0
    coverages: list[float] = []
    risks: list[float] = []

    for _ in range(n_splits):
        order = rng.permutation(count)
        calibration, test = order[:n_cal], order[n_cal:]

        result = certify(score_array[calibration], error_array[calibration], alpha, delta, grid)
        if not result.certified:
            continue

        certified += 1
        accepted = score_array[test] >= result.threshold
        if not accepted.any():
            coverages.append(0.0)
            risks.append(0.0)
            continue

        realised = float(error_array[test][accepted].mean())
        coverages.append(float(accepted.mean()))
        risks.append(realised)
        if realised > alpha:
            violations += 1

    return ValiditySummary(
        alpha=alpha,
        delta=delta,
        n_splits=n_splits,
        certified_rate=certified / n_splits,
        violation_rate=violations / n_splits,
        mean_coverage=float(np.mean(coverages)) if coverages else 0.0,
        sd_coverage=float(np.std(coverages)) if coverages else 0.0,
        mean_realised_risk=float(np.mean(risks)) if risks else 0.0,
        sd_realised_risk=float(np.std(risks)) if risks else 0.0,
        p95_realised_risk=float(np.percentile(risks, 95)) if risks else 0.0,
        n_certified=certified,
    )


def evaluate_methods(
    scored: pd.DataFrame,
    methods: Sequence[str],
    alphas: Sequence[float],
    delta: float,
    *,
    n_splits: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Resplit validity for every method and error budget on the pooled rows."""
    pooled = scored[scored["split"].isin(OUT_OF_SAMPLE_SPLITS)]
    if pooled.empty:
        raise ValueError("no calibration or test rows to pool")

    errors = 1 - pooled["correct"].to_numpy(dtype=int)
    rows: list[dict[str, float | str]] = []
    for alpha in alphas:
        for method in methods:
            summary = resplit_validity(
                pooled[method].to_numpy(dtype=float),
                errors,
                alpha,
                delta,
                n_splits=n_splits,
                seed=seed,
            )
            rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "delta": delta,
                    "certified_rate": summary.certified_rate,
                    "violation_rate": summary.violation_rate,
                    "n_certified": summary.n_certified,
                    "conditional_violation_rate": summary.conditional_violation_rate,
                    "mean_coverage": summary.mean_coverage,
                    "sd_coverage": summary.sd_coverage,
                    "mean_realised_risk": summary.mean_realised_risk,
                    "sd_realised_risk": summary.sd_realised_risk,
                    "p95_realised_risk": summary.p95_realised_risk,
                }
            )
    return pd.DataFrame(rows)

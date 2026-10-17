"""Risk-coverage evaluation.

A selective predictor accepts the most confident items first. The risk-coverage
curve plots the error rate among accepted items against the fraction accepted, so
it answers the question this project is built on: *how much can be auto-processed
before the error rate exceeds a target?*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class RiskCoverage:
    """Risk at each achievable coverage level, from most to least confident."""

    coverage: np.ndarray
    risk: np.ndarray

    def __len__(self) -> int:
        return int(self.coverage.shape[0])


def risk_coverage_curve(scores: Sequence[float], errors: Sequence[int]) -> RiskCoverage:
    """Risk-coverage curve, ordering items by descending score."""
    score_array = np.asarray(scores, dtype=float)
    error_array = np.asarray(errors, dtype=float)
    if score_array.shape != error_array.shape:
        raise ValueError("scores and errors must have the same length")
    if score_array.size == 0:
        raise ValueError("cannot build a curve from no items")

    order = np.argsort(-score_array, kind="stable")
    cumulative_errors = np.cumsum(error_array[order])
    accepted = np.arange(1, score_array.size + 1)

    return RiskCoverage(coverage=accepted / score_array.size, risk=cumulative_errors / accepted)


def auroc(scores: Sequence[float], errors: Sequence[int]) -> float:
    """Area under the ROC curve: how well the score ranks correct above incorrect."""
    labels = np.asarray(errors, dtype=int)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(1 - labels, scores))


def aurc(scores: Sequence[float], errors: Sequence[int]) -> float:
    """Area under the risk-coverage curve (lower is better).

    Coverage advances in equal steps of ``1/n``, so the area is exactly the mean of
    the risk values along the curve.
    """
    return float(np.mean(risk_coverage_curve(scores, errors).risk))


def risk_at_coverage(scores: Sequence[float], errors: Sequence[int], coverage: float) -> float:
    """Error rate among the most confident ``coverage`` fraction of items."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    curve = risk_coverage_curve(scores, errors)
    index = max(0, min(len(curve) - 1, int(np.ceil(coverage * len(curve))) - 1))
    return float(curve.risk[index])


def coverage_at_risk(
    scores: Sequence[float], errors: Sequence[int], max_risk: float
) -> float:
    """Largest coverage whose empirical risk stays at or below ``max_risk``."""
    curve = risk_coverage_curve(scores, errors)
    acceptable = np.nonzero(curve.risk <= max_risk)[0]
    if acceptable.size == 0:
        return 0.0
    return float(curve.coverage[acceptable[-1]])


def summarise(
    scores: Sequence[float], errors: Sequence[int], target_risks: Sequence[float] = (0.02, 0.05, 0.10)
) -> dict[str, float]:
    """AUROC, AURC and coverage achievable at each target risk."""
    summary = {"auroc": auroc(scores, errors), "aurc": aurc(scores, errors)}
    for target in target_risks:
        summary[f"coverage@{target:.2f}"] = coverage_at_risk(scores, errors, target)
    return summary

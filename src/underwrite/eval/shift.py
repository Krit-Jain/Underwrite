"""Distribution shift: calibrating on one corpus and deploying on another.

The certificate is only valid while new documents are exchangeable with the
calibration documents. Pointing a CORD-calibrated gate at Indian GST invoices breaks
that assumption, and this module measures the damage and what it costs to repair.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from underwrite.conformal.ltt import certify


def calibrate_and_deploy(
    source_scores: Sequence[float],
    source_errors: Sequence[int],
    target_scores: Sequence[float],
    target_errors: Sequence[int],
    alpha: float,
    delta: float,
    grid: Sequence[float] | None = None,
) -> dict[str, object]:
    """Certify on the source corpus, then apply the threshold to the target corpus."""
    result = certify(source_scores, source_errors, alpha, delta, grid)
    row: dict[str, object] = {
        "alpha": alpha,
        "delta": delta,
        "certified": result.certified,
        "threshold": result.threshold,
        "source_coverage": result.coverage,
        "source_risk": result.selective_risk,
        "target_coverage": 0.0,
        "target_risk": float("nan"),
        "violated": False,
    }
    if not result.certified:
        return row

    scores = np.asarray(target_scores, dtype=float)
    errors = np.asarray(target_errors, dtype=float)
    accepted = scores >= result.threshold
    row["target_coverage"] = float(accepted.mean())
    if accepted.any():
        risk = float(errors[accepted].mean())
        row["target_risk"] = risk
        row["violated"] = bool(risk > alpha)
    return row


def shift_table(
    source: pd.DataFrame,
    target: pd.DataFrame,
    methods: Sequence[str],
    alphas: Sequence[float],
    delta: float,
    grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Calibrate on ``source`` and deploy on ``target`` for every method and budget."""
    source_errors = 1 - source["correct"].to_numpy(dtype=int)
    target_errors = 1 - target["correct"].to_numpy(dtype=int)

    rows = []
    for alpha in alphas:
        for method in methods:
            row = calibrate_and_deploy(
                source[method].to_numpy(dtype=float),
                source_errors,
                target[method].to_numpy(dtype=float),
                target_errors,
                alpha,
                delta,
                grid,
            )
            row["method"] = method
            rows.append(row)
    return pd.DataFrame(rows)


def repair_curve(
    source_scores: Sequence[float],
    source_errors: Sequence[int],
    target_scores: Sequence[float],
    target_errors: Sequence[int],
    ks: Sequence[int],
    alpha: float,
    delta: float,
    *,
    n_draws: int = 50,
    seed: int = 0,
    grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Recalibrate with ``k`` labelled target documents added to the source calibration.

    Each draw labels a random ``k`` of the target documents, recalibrates, and scores
    the remaining target documents -- so the added labels are never used to grade the
    documents they came from.
    """
    source_scores = np.asarray(source_scores, dtype=float)
    source_errors = np.asarray(source_errors, dtype=float)
    target_scores = np.asarray(target_scores, dtype=float)
    target_errors = np.asarray(target_errors, dtype=float)
    if source_scores.shape != source_errors.shape:
        raise ValueError("source scores and errors must have the same length")
    if target_scores.shape != target_errors.shape:
        raise ValueError("target scores and errors must have the same length")

    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        if k >= target_scores.size:
            raise ValueError("k must be smaller than the target corpus")
        certified = 0
        violations = 0
        coverages: list[float] = []
        risks: list[float] = []

        for _ in range(n_draws):
            order = rng.permutation(target_scores.size)
            labelled, heldout = order[:k], order[k:]

            calibration_scores = np.concatenate([source_scores, target_scores[labelled]])
            calibration_errors = np.concatenate([source_errors, target_errors[labelled]])

            result = certify(calibration_scores, calibration_errors, alpha, delta, grid)
            if not result.certified:
                continue

            certified += 1
            accepted = target_scores[heldout] >= result.threshold
            if not accepted.any():
                coverages.append(0.0)
                risks.append(0.0)
                continue

            risk = float(target_errors[heldout][accepted].mean())
            coverages.append(float(accepted.mean()))
            risks.append(risk)
            if risk > alpha:
                violations += 1

        rows.append(
            {
                "k": k,
                "alpha": alpha,
                "n_draws": n_draws,
                "certified_rate": certified / n_draws,
                "violation_rate": violations / n_draws,
                "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
                "sd_coverage": float(np.std(coverages)) if coverages else 0.0,
                "mean_risk": float(np.mean(risks)) if risks else float("nan"),
            }
        )
    return pd.DataFrame(rows)

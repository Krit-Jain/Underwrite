"""Audit lane: unbiased error monitoring on auto-accepted items.

Labels are only available on items a human looked at, and humans look at *flagged*
items -- which are the ones the system already suspects. Any error rate estimated
from flagged items alone is therefore selection-biased and says nothing about the
items that were auto-accepted.

The audit lane fixes that by sampling a small random slice of **auto-accepted** items
and checking those. From the audited sample we maintain a Clopper-Pearson upper
confidence bound on the error rate among accepted items, and raise an alarm when the
bound stops supporting the certified ``alpha`` -- the signal that the exchangeability
assumption has broken and the certificate needs recalibrating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

from underwrite.conformal.ltt import minimum_accepts_for_zero_errors

DEFAULT_CONFIDENCE = 0.90


def clopper_pearson_upper(
    errors: int, total: int, confidence: float = DEFAULT_CONFIDENCE
) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial error rate."""
    if total <= 0:
        return 1.0
    if errors < 0 or errors > total:
        raise ValueError("errors must lie between 0 and total")
    if errors == total:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, total - errors))


@dataclass
class AuditResult:
    """Trajectory of the audit lane over a stream."""

    positions: list[int] = field(default_factory=list)
    audited: list[int] = field(default_factory=list)
    errors: list[int] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "position": self.positions,
                "audited": self.audited,
                "errors": self.errors,
                "upper_bound": self.upper_bounds,
            }
        )

    def first_alarm(self, alpha: float) -> int | None:
        """Position of the first check whose upper bound exceeded ``alpha``."""
        for position, bound in zip(self.positions, self.upper_bounds):
            if bound > alpha:
                return position
        return None


def simulate_audit(
    stream_errors: Sequence[int],
    *,
    audit_rate: float = 0.10,
    alpha: float = 0.10,
    confidence: float = DEFAULT_CONFIDENCE,
    check_every: int = 10,
    minimum_audits: int | None = None,
    seed: int = 0,
) -> AuditResult:
    """Sample a random slice of accepted items and track the error bound.

    ``stream_errors`` is the *true* correctness of each accepted item in stream
    order; an entry is only observed when that item happens to be audited.

    A check is only recorded once ``minimum_audits`` items have been audited. Below
    that, an error-free sample still cannot support ``alpha`` -- the bound is
    necessarily above it -- so reporting there would raise an alarm on every stream,
    no matter how clean.
    """
    if not 0.0 < audit_rate <= 1.0:
        raise ValueError("audit_rate must lie in (0, 1]")

    floor = (
        minimum_accepts_for_zero_errors(alpha, 1.0 - confidence)
        if minimum_audits is None
        else minimum_audits
    )

    rng = np.random.default_rng(seed)
    result = AuditResult()
    audited = 0
    errors = 0

    for position, is_error in enumerate(stream_errors, start=1):
        if rng.random() >= audit_rate:
            continue
        audited += 1
        errors += int(is_error)
        if audited >= floor and audited % check_every == 0:
            result.positions.append(position)
            result.audited.append(audited)
            result.errors.append(errors)
            result.upper_bounds.append(clopper_pearson_upper(errors, audited, confidence))

    return result


def injected_drift(
    clean_length: int, drift_length: int, clean_error: float, drift_error: float, seed: int = 0
) -> np.ndarray:
    """A stream that is well-behaved and then degrades at ``clean_length``."""
    rng = np.random.default_rng(seed)
    clean = (rng.random(clean_length) < clean_error).astype(int)
    drifted = (rng.random(drift_length) < drift_error).astype(int)
    return np.concatenate([clean, drifted])

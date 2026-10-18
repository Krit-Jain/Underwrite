"""Audit lane: unbiased drift detection on auto-accepted items.

Labels are only available on items a human looked at, and humans look at *flagged*
items -- the ones the system already distrusts. Any error rate estimated from flagged
items alone is selection-biased and says nothing about the auto-accepted ones, so the
audit lane samples a random slice of **auto-accepted** items and checks those.

The alarm asks one specific question: *is the true accepted-error rate above the
certified alpha?* Answering it needs the Clopper-Pearson **lower** bound to exceed
alpha. An upper bound above alpha only says "not enough evidence yet" -- alarming on
that fires on almost every clean stream, because with few audits the bound is
necessarily wide.

Because the bound is re-tested at every check, each look runs at a confidence
corrected for the number of looks (Bonferroni), so the probability of *any* false
alarm across a stream stays inside a chosen budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

DEFAULT_FALSE_ALARM_BUDGET = 0.05


def clopper_pearson_lower(errors: int, total: int, confidence: float = 0.95) -> float:
    """One-sided lower bound on a binomial rate; ``0.0`` when no errors were seen."""
    if total <= 0 or errors <= 0:
        return 0.0
    if errors > total:
        raise ValueError("errors must lie between 0 and total")
    return float(beta.ppf(1.0 - confidence, errors, total - errors + 1))


def clopper_pearson_upper(errors: int, total: int, confidence: float = 0.95) -> float:
    """One-sided upper bound, reported alongside for how tight the estimate is."""
    if total <= 0:
        return 1.0
    if errors < 0 or errors > total:
        raise ValueError("errors must lie between 0 and total")
    if errors == total:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, total - errors))


def per_look_confidence(false_alarm_budget: float, looks: int) -> float:
    """Confidence for one look so that ``looks`` looks keep the budget."""
    if not 0.0 < false_alarm_budget < 1.0:
        raise ValueError("false_alarm_budget must lie in (0, 1)")
    if looks < 1:
        raise ValueError("looks must be at least 1")
    return 1.0 - false_alarm_budget / looks


@dataclass
class AuditResult:
    """Trajectory of the audit lane over a stream."""

    positions: list[int] = field(default_factory=list)
    audited: list[int] = field(default_factory=list)
    errors: list[int] = field(default_factory=list)
    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    per_look_confidence: float = 0.0

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "position": self.positions,
                "audited": self.audited,
                "errors": self.errors,
                "lower_bound": self.lower_bounds,
                "upper_bound": self.upper_bounds,
            }
        )

    def first_alarm(self, alpha: float) -> int | None:
        """Position of the first check whose lower bound exceeded ``alpha``."""
        for position, bound in zip(self.positions, self.lower_bounds):
            if bound > alpha:
                return position
        return None


def simulate_audit(
    stream_errors: Sequence[int],
    *,
    audit_rate: float = 0.10,
    alpha: float = 0.10,
    false_alarm_budget: float = DEFAULT_FALSE_ALARM_BUDGET,
    check_every: int = 10,
    looks: int | None = None,
    seed: int = 0,
) -> AuditResult:
    """Sample a random slice of accepted items and watch the lower bound.

    ``stream_errors`` is the true correctness of each accepted item in stream order;
    an entry is only observed when that item happens to be audited.

    ``seed`` seeds the *audit selection* only. Pass a seed that is independent of the
    one used to generate the stream: reuse it and the audit decisions replay the same
    random draws as the data, which correlates the sample with the errors and silently
    produces a biased estimate.
    """
    if not 0.0 < audit_rate <= 1.0:
        raise ValueError("audit_rate must lie in (0, 1]")

    if looks is None:
        expected_audits = max(1, int(len(stream_errors) * audit_rate))
        looks = max(1, expected_audits // max(1, check_every))
    confidence = per_look_confidence(false_alarm_budget, looks)

    rng = np.random.default_rng(seed)
    result = AuditResult(per_look_confidence=confidence)
    audited = 0
    errors = 0

    for position, is_error in enumerate(stream_errors, start=1):
        if rng.random() >= audit_rate:
            continue
        audited += 1
        errors += int(is_error)
        if audited % check_every == 0:
            result.positions.append(position)
            result.audited.append(audited)
            result.errors.append(errors)
            result.lower_bounds.append(clopper_pearson_lower(errors, audited, confidence))
            result.upper_bounds.append(clopper_pearson_upper(errors, audited, confidence))

    return result


def injected_drift(
    clean_length: int,
    drift_length: int,
    clean_error: float,
    drift_error: float,
    seed: int = 0,
) -> np.ndarray:
    """A stream that behaves, then degrades at ``clean_length``."""
    rng = np.random.default_rng(seed)
    clean = (rng.random(clean_length) < clean_error).astype(int)
    drifted = (rng.random(drift_length) < drift_error).astype(int)
    return np.concatenate([clean, drifted])

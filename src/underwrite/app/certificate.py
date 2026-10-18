"""The batch certificate: what is actually being promised, in plain language.

Everything here is arithmetic over per-field decisions. Nothing is estimated or
rounded in the prose: the numbers in the caveat are the numbers in the record.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

DECISION_AUTO = "auto"
DECISION_REVIEW = "review"


@dataclass(frozen=True)
class Certificate:
    """A signed-off summary of one batch."""

    alpha: float
    delta: float
    threshold: float | None
    model_id: str
    calibration_documents: int
    counts: Mapping[str, int]
    audit_sample: int = 0

    @property
    def certified(self) -> bool:
        return self.threshold is not None

    @property
    def total_fields(self) -> int:
        return sum(self.counts.values())

    @property
    def auto_fields(self) -> int:
        return int(self.counts.get(DECISION_AUTO, 0))

    @property
    def review_fields(self) -> int:
        return int(self.counts.get(DECISION_REVIEW, 0))

    @property
    def auto_rate(self) -> float:
        return self.auto_fields / self.total_fields if self.total_fields else 0.0

    def caveat(self) -> str:
        """The guarantee, stated together with the assumption it rests on."""
        if not self.certified:
            return (
                "This batch is not certified. With the calibration data available, no "
                f"accept threshold could hold the per-field error rate at or below "
                f"{self.alpha:.0%} with {1 - self.delta:.0%} confidence, so every field "
                "was routed to review."
            )
        return (
            f"With {1 - self.delta:.0%} confidence, the error rate among the "
            f"{self.auto_fields} auto-accepted fields is at most {self.alpha:.0%}. "
            f"The threshold was set on {self.calibration_documents} labelled documents, "
            f"and {self.auto_rate:.0%} of this batch cleared it. That promise assumes "
            "these documents resemble the calibration ones. If they don't -- a new "
            "supplier layout, a different country, a changed template -- it can silently "
            f"fail, which is why {self.audit_sample} auto-accepted fields are randomly "
            "sampled for checking rather than trusting the certificate blindly."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "certified": self.certified,
            "alpha": self.alpha,
            "delta": self.delta,
            "threshold": self.threshold,
            "model_id": self.model_id,
            "calibration_documents": self.calibration_documents,
            "counts": dict(self.counts),
            "total_fields": self.total_fields,
            "auto_fields": self.auto_fields,
            "review_fields": self.review_fields,
            "auto_rate": self.auto_rate,
            "audit_sample": self.audit_sample,
            "caveat": self.caveat(),
        }


def build_certificate(
    decisions: Sequence[str],
    *,
    alpha: float,
    delta: float,
    threshold: float | None,
    model_id: str,
    calibration_documents: int,
    audit_sample: int = 0,
) -> Certificate:
    """Summarise a batch's per-field decisions."""
    unknown = set(decisions) - {DECISION_AUTO, DECISION_REVIEW}
    if unknown:
        raise ValueError(f"unknown decisions: {sorted(unknown)}")
    return Certificate(
        alpha=alpha,
        delta=delta,
        threshold=threshold,
        model_id=model_id,
        calibration_documents=calibration_documents,
        counts=dict(Counter(decisions)),
        audit_sample=audit_sample,
    )

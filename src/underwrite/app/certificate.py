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
        """The guarantee, stated with the assumption it rests on."""
        if not self.certified:
            return (
                "This batch is not certified. With the calibration data available, no "
                f"accept threshold could hold the per-field error rate at or below "
                f"{self.alpha:.0%} with {1 - self.delta:.0%} confidence, so every field "
                "was routed to review."
            )
        return (
            f"With {1 - self.delta:.0%} confidence, the long-run error rate on documents "
            f"like the {self.calibration_documents} labelled ones used for calibration is "
            f"at most {self.alpha:.0%}. {self.auto_rate:.0%} of this batch cleared the "
            "threshold; that describes this batch, not the guarantee -- the promise is "
            "about the population of similar documents, and it holds only while new "
            "documents resemble the calibration ones. A new supplier layout, a different "
            "country or a changed template can break it silently, which is why "
            f"{self.audit_sample} auto-accepted fields are randomly sampled for checking."
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

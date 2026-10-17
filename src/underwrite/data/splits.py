"""Document-level, seeded dataset splits.

Splits are assigned per document and never per field, so every field of a document
lands in exactly one split. Assignment is deterministic for a given seed.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

SPLIT_NAMES = ("probe_train", "calibration", "test")


class SplitError(ValueError):
    """Raised when split fractions or names are invalid."""


def assign_splits(
    doc_ids: Iterable[str],
    *,
    probe_train: float = 0.4,
    calibration: float = 0.3,
    test: float = 0.3,
    seed: int = 0,
) -> dict[str, str]:
    """Assign each document id to ``probe_train``, ``calibration`` or ``test``."""
    fractions = (probe_train, calibration, test)
    if any(fraction < 0.0 for fraction in fractions):
        raise SplitError("split fractions must be non-negative")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise SplitError(f"split fractions must sum to 1.0 (got {sum(fractions):.6f})")

    ids = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    count = len(ids)
    train_end = round(count * probe_train)
    calibration_end = train_end + round(count * calibration)

    assignment: dict[str, str] = {}
    for index, doc_id in enumerate(ids):
        if index < train_end:
            split = "probe_train"
        elif index < calibration_end:
            split = "calibration"
        else:
            split = "test"
        assignment[doc_id] = split
    return assignment


def summarize(assignment: Mapping[str, str]) -> dict[str, int]:
    """Count the documents in each split."""
    counts = Counter(assignment.values())
    return {name: counts.get(name, 0) for name in SPLIT_NAMES}


def write_splits(path: str | Path, assignment: Mapping[str, str]) -> Path:
    """Write a split assignment to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "splits": dict(sorted(assignment.items())),
        "counts": summarize(assignment),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_splits(path: str | Path) -> dict[str, str]:
    """Read a split assignment written by :func:`write_splits`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignment = payload.get("splits")
    if not isinstance(assignment, dict):
        raise SplitError(f"invalid splits file: {path}")
    return assignment

"""Per-document feature bundles for the probe and the fusion model.

A document's bundle holds, for every configured field:

* the handful of token-level statistics (log-probabilities, entropy, margin),
* the hidden-state vector at the **last** value token, and the **mean** over the
  value tokens, for each probe layer.

Bundles are written as one compressed ``.npz`` per document so a long extraction run
is resumable and its features can be loaded without re-running the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from underwrite.extract.vlm import TeacherForcedOutput
from underwrite.signals.teacher_forced import (
    FEATURE_NAMES,
    field_statistics,
    locate_field_values,
    select_tokens,
)
from underwrite.signals.validators import CHECK_NAMES

HIDDEN_KINDS = ("last", "mean")
META_KEY = "__meta__"


@dataclass(frozen=True)
class FieldFeatures:
    """Signals for one field of one document."""

    token: dict[str, float]
    # layer index -> array of shape (len(HIDDEN_KINDS), hidden size)
    hidden: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class DocumentFeatures:
    """Everything captured for one document."""

    doc_id: str
    fields: dict[str, FieldFeatures]
    validators: dict[str, float]
    extracted: dict[str, Any]
    gold: dict[str, str]

    def token_row(self, name: str) -> list[float]:
        summary = self.fields.get(name)
        if summary is None:
            return [math.nan] * len(FEATURE_NAMES)
        return [summary.token.get(feature, math.nan) for feature in FEATURE_NAMES]


def build_document_features(
    doc_id: str,
    extracted: Mapping[str, Any],
    gold: Mapping[str, str],
    signals: TeacherForcedOutput,
    fields: Sequence[str],
    validators: Mapping[str, float] | None = None,
) -> DocumentFeatures:
    """Assemble one document's bundle from a teacher-forced output."""
    spans = locate_field_values(signals.text, fields)
    captured: dict[str, FieldFeatures] = {}

    for name in fields:
        span = spans.get(name)
        if span is None:
            continue
        indices = select_tokens(signals.offsets, span)
        if not indices:
            continue

        summary = field_statistics(
            [signals.logprobs[index] for index in indices],
            [signals.entropies[index] for index in indices],
            [signals.margins[index] for index in indices],
        )
        if summary is None:
            continue

        hidden: dict[int, np.ndarray] = {}
        for layer, matrix in signals.hidden_states.items():
            rows = matrix[indices]
            hidden[layer] = np.stack([rows[-1], rows.mean(axis=0)]).astype(np.float16)

        captured[name] = FieldFeatures(token=summary.as_features(), hidden=hidden)

    return DocumentFeatures(
        doc_id=doc_id,
        fields=captured,
        validators=dict(validators or {}),
        extracted=dict(extracted),
        gold=dict(gold),
    )


def save_features(
    path: str | Path,
    features: DocumentFeatures,
    fields: Sequence[str],
    layers: Sequence[int],
) -> Path:
    """Write one document's bundle as a compressed ``.npz``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for name in fields:
        arrays[f"token__{name}"] = np.asarray(
            features.token_row(name), dtype=np.float32
        )
        captured = features.fields.get(name)
        for layer in layers:
            if captured is not None and layer in captured.hidden:
                key = f"hidden__{name}__L{layer}"
                arrays[key] = captured.hidden[layer]

    arrays["validators"] = np.asarray(
        [features.validators.get(check, 0.0) for check in CHECK_NAMES], dtype=np.float32
    )
    arrays[META_KEY] = np.asarray(
        json.dumps(
            {
                "doc_id": features.doc_id,
                "fields": list(fields),
                "layers": list(layers),
                "feature_names": list(FEATURE_NAMES),
                "check_names": list(CHECK_NAMES),
                "hidden_kinds": list(HIDDEN_KINDS),
                "extracted": features.extracted,
                "gold": features.gold,
            }
        )
    )
    np.savez_compressed(target, **arrays)
    return target


def load_features(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a bundle back, returning its arrays and decoded metadata."""
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != META_KEY}
        meta = json.loads(str(archive[META_KEY])) if META_KEY in archive.files else {}
    return arrays, meta

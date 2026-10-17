"""Run the whole capture path for a single document.

Shared by the dry-run harness and the resumable extraction run so the two cannot
drift apart: both go through :func:`capture_document`.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

from underwrite.extract.parse import ParseError, extract_json_object
from underwrite.extract.vlm import build_messages, generate, teacher_forced
from underwrite.probe.features import DocumentFeatures, build_document_features
from underwrite.signals.validators import validate_document
from underwrite.signals.verbalized import ask_verbalized_confidence


@dataclass
class CapturedDocument:
    """Everything one document yields: text, fields, checks, features, timings."""

    doc_id: str
    text: str
    extracted: dict[str, Any]
    validators: dict[str, float]
    features: DocumentFeatures
    generation_seconds: float
    n_prompt_tokens: int
    n_response_tokens: int
    offsets_aligned: bool
    verbalized: dict[str, float]

    def trace_record(self, source: str) -> dict[str, Any]:
        """A JSON-serialisable summary for the trace log."""
        return {
            "doc_id": self.doc_id,
            "source": source,
            "generation_seconds": self.generation_seconds,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_response_tokens": self.n_response_tokens,
            "offsets_aligned": self.offsets_aligned,
            "extracted": self.extracted,
            "gold": self.features.gold,
            "validators": self.validators,
            "verbalized": self.verbalized,
            "fields": {
                name: captured.token for name, captured in self.features.fields.items()
            },
            "hidden_shapes": {
                str(layer): list(matrix.shape)
                for captured in self.features.fields.values()
                for layer, matrix in captured.hidden.items()
            },
        }


def capture_document(
    processor: Any,
    model: Any,
    doc_id: str,
    image: Any,
    gold: Mapping[str, str],
    fields: Sequence[str],
    layers: Sequence[int],
    *,
    max_new_tokens: int = 256,
    with_verbalized: bool = False,
) -> CapturedDocument:
    """Extract, validate, and capture signals for one document."""
    messages = build_messages(image, fields)

    started = perf_counter()
    text = generate(processor, model, messages, max_new_tokens)
    generation_seconds = perf_counter() - started

    try:
        extracted = extract_json_object(text)
    except ParseError:
        extracted = {}

    signals = teacher_forced(processor, model, messages, text, layers)
    validators = validate_document(extracted)
    features = build_document_features(doc_id, extracted, gold, signals, fields, validators)

    verbalized = (
        ask_verbalized_confidence(processor, model, messages, text, fields)
        if with_verbalized
        else {}
    )

    return CapturedDocument(
        doc_id=doc_id,
        text=text,
        extracted=extracted,
        validators=validators,
        features=features,
        generation_seconds=generation_seconds,
        n_prompt_tokens=signals.n_prompt_tokens,
        n_response_tokens=len(signals.token_ids),
        offsets_aligned=signals.offsets_aligned,
        verbalized=verbalized,
    )

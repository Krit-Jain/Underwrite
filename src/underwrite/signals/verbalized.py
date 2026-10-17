"""Verbalized-confidence baseline: ask the model to rate its own extraction.

Published work on CORD reports verbalized (and single-sample consensus) confidence as
close to uninformative. It is kept here as the ``S1`` baseline so that claim can be
measured on this pipeline rather than assumed.
"""

from __future__ import annotations

from typing import Any, Sequence

from underwrite.extract.parse import ParseError, extract_json_object

VERBALIZED_TEMPLATE = (
    "Rate your own previous answer. For each field below, give the probability between "
    "0 and 1 that the value you returned is correct.\n"
    "Fields: {fields}\n"
    "Reply with a single JSON object mapping each field name to a number. No prose, "
    "no code fences."
)


def verbalized_instruction(fields: Sequence[str]) -> str:
    """The follow-up user turn asking for self-reported probabilities."""
    return VERBALIZED_TEMPLATE.format(fields=", ".join(fields))


def parse_verbalized(text: str, fields: Sequence[str]) -> dict[str, float]:
    """Per-field probabilities recovered from a self-rating response.

    Values are clamped to ``[0, 1]``. Booleans, strings and unknown fields are
    ignored rather than coerced, so a malformed rating simply drops out.
    """
    try:
        raw = extract_json_object(text)
    except ParseError:
        return {}

    scores: dict[str, float] = {}
    for name in fields:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        scores[name] = min(1.0, max(0.0, float(value)))
    return scores


def ask_verbalized_confidence(
    processor: Any,
    model: Any,
    messages: Sequence[dict[str, Any]],
    response_text: str,
    fields: Sequence[str],
    max_new_tokens: int = 160,
) -> dict[str, float]:
    """Run one extra cheap call asking the model to score its own answer."""
    from underwrite.extract.vlm import generate

    conversation = list(messages) + [
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": verbalized_instruction(fields)},
    ]
    text = generate(processor, model, conversation, max_new_tokens)
    return parse_verbalized(text, fields)

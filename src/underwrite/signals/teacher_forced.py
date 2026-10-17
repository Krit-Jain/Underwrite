"""Teacher-forced signal capture: per-field signals from the model's own distribution.

After generation, the same (image, prompt, generated JSON) sequence is forwarded
once more with ``output_hidden_states=True``. For every field we recover the tokens
that spell its value and summarise them into black-box signals (realised
log-probabilities, entropy, top-1/top-2 margin) plus hidden-state vectors for the
probe.

Everything in this module that does not need the network is a pure function, so the
alignment logic is unit-testable without loading a model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

# Stable ordering so downstream feature vectors are reproducible.
FEATURE_NAMES = (
    "mean_logprob",
    "min_logprob",
    "first_token_entropy",
    "first_token_margin",
    "n_tokens",
)


def _end_of_string(text: str, opening: int) -> int | None:
    """Index just past the closing quote, or ``None`` if the string is unterminated."""
    index = opening + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    return None


def locate_field_values(text: str, fields: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Character spans of each field's *value* inside ``text``.

    String values exclude their surrounding quotes; numeric and bare literals are
    trimmed of trailing whitespace. Fields that do not appear are omitted.
    """
    spans: dict[str, tuple[int, int]] = {}
    for field in fields:
        match = re.search(r'"%s"\s*:\s*' % re.escape(field), text)
        if match is None:
            continue
        start = match.end()
        if start >= len(text):
            continue

        if text[start] == '"':
            closing = _end_of_string(text, start)
            # An unterminated string runs to the end of the text; do not drop its
            # final character, which subtracting one for the closing quote would.
            inner = (start + 1, len(text) if closing is None else closing - 1)
            if inner[1] > inner[0]:
                spans[field] = inner
        else:
            end = start
            while end < len(text) and text[end] not in ",}\n\r":
                end += 1
            trimmed = text[start:end].rstrip()
            if trimmed:
                spans[field] = (start, start + len(trimmed))
    return spans


def select_tokens(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    """Indices of tokens whose character range overlaps ``span``.

    Special tokens report a ``(0, 0)`` offset and are never selected.
    """
    low, high = span
    return [index for index, (start, end) in enumerate(offsets) if end > low and start < high]


@dataclass(frozen=True)
class TokenStatistics:
    """Signals summarising one field's value tokens."""

    mean_logprob: float
    min_logprob: float
    first_token_entropy: float
    first_token_margin: float
    n_tokens: int

    def as_features(self) -> dict[str, float]:
        return {
            "mean_logprob": self.mean_logprob,
            "min_logprob": self.min_logprob,
            "first_token_entropy": self.first_token_entropy,
            "first_token_margin": self.first_token_margin,
            "n_tokens": float(self.n_tokens),
        }


def field_statistics(
    logprobs: Sequence[float],
    entropies: Sequence[float],
    margins: Sequence[float],
) -> TokenStatistics | None:
    """Summarise one field's value tokens, or ``None`` when it has none."""
    if not logprobs:
        return None
    return TokenStatistics(
        mean_logprob=sum(logprobs) / len(logprobs),
        min_logprob=min(logprobs),
        first_token_entropy=entropies[0],
        first_token_margin=margins[0],
        n_tokens=len(logprobs),
    )


def entropy(log_probs: Sequence[float]) -> float:
    """Shannon entropy in nats of a log-probability vector."""
    total = 0.0
    for value in log_probs:
        if value == -math.inf:
            continue
        total -= value * math.exp(value)
    return total


def top_margin(log_probs: Sequence[float]) -> float:
    """Difference between the two largest log-probabilities."""
    if len(log_probs) < 2:
        return 0.0
    first, second = sorted(log_probs, reverse=True)[:2]
    return first - second


def summarise_fields(
    text: str,
    offsets: Sequence[tuple[int, int]],
    logprobs: Sequence[float],
    entropies: Sequence[float],
    margins: Sequence[float],
    fields: Sequence[str],
) -> dict[str, TokenStatistics]:
    """Per-field statistics for every value token that the tokenizer covers."""
    statistics: dict[str, TokenStatistics] = {}
    for field, span in locate_field_values(text, fields).items():
        indices = select_tokens(offsets, span)
        if not indices:
            continue
        summary = field_statistics(
            [logprobs[index] for index in indices],
            [entropies[index] for index in indices],
            [margins[index] for index in indices],
        )
        if summary is not None:
            statistics[field] = summary
    return statistics


def flatten(statistics: Mapping[str, TokenStatistics]) -> dict[str, float]:
    """Flatten per-field statistics into a ``field.feature`` -> value mapping."""
    flat: dict[str, float] = {}
    for field, summary in statistics.items():
        for name, value in summary.as_features().items():
            flat[f"{field}.{name}"] = value
    return flat


def as_array(
    statistics: Mapping[str, TokenStatistics], fields: Sequence[str]
) -> dict[str, list[float]]:
    """Build one numeric row per field in a fixed order (missing fields -> NaN)."""
    rows: dict[str, list[float]] = {}
    for field in fields:
        summary = statistics.get(field)
        if summary is None:
            rows[field] = [math.nan] * len(FEATURE_NAMES)
        else:
            features = summary.as_features()
            rows[field] = [features[name] for name in FEATURE_NAMES]
    return rows

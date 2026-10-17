"""Source-grounding support: does a decoded value actually appear on the page?

This is a *black-box* signal -- it needs only the document's text, never the model's
internals. Published work on CORD finds grounding to be the strongest black-box
signal, so it is a required baseline here rather than an optional extra.

Money values are located by exact amount (a page token must parse to the same
number); free text is located by fuzzy substring match. Dates are left unsupported,
because a model that returns ISO dates will never literally match a receipt's own
date formatting -- scoring that would only add noise.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Iterable, Sequence

from rapidfuzz import fuzz

from underwrite.data.normalize import MONEY_FIELDS, normalize_vendor, parse_money

TEXT_FIELDS = ("vendor",)
UNSUPPORTED_FIELDS = ("date",)

_DIGITS = re.compile(r"[^0-9]")


def page_tokens(ground_truth: str | dict[str, Any]) -> list[str]:
    """OCR word strings from a CORD ground-truth blob (or an already-parsed dict)."""
    parsed = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    tokens: list[str] = []
    for line in parsed.get("valid_line") or []:
        for word in line.get("words") or []:
            text = word.get("text")
            if isinstance(text, str) and text.strip():
                tokens.append(text.strip())
    return tokens


def _digits(value: Any) -> str:
    return _DIGITS.sub("", str(value))


def numeric_support(
    value: Any, tokens: Sequence[str], *, decimal_sep: str | None = None
) -> float:
    """``1.0`` when a page token parses to exactly ``value``, otherwise ``0.0``."""
    target = parse_money(value, decimal_sep=decimal_sep)
    if target is None:
        return 0.0
    for token in tokens:
        candidate = parse_money(token, decimal_sep=decimal_sep)
        if candidate is not None and candidate == target:
            return 1.0
    return 0.0


def text_support(value: Any, tokens: Sequence[str]) -> float:
    """Graded support for free text, from the best substring match on the page."""
    target = normalize_vendor(value)
    if not target:
        return 0.0
    page = " ".join(normalize_vendor(token) for token in tokens).strip()
    if not page:
        return 0.0
    return fuzz.partial_ratio(target, page) / 100.0


def grounding_support(
    field: str,
    value: Any,
    tokens: Sequence[str],
    *,
    decimal_sep: str | None = None,
) -> float | None:
    """Support score for one field, or ``None`` when grounding does not apply."""
    if value is None:
        return 0.0
    if field in MONEY_FIELDS:
        return numeric_support(value, tokens, decimal_sep=decimal_sep)
    if field in TEXT_FIELDS:
        return text_support(value, tokens)
    if field in UNSUPPORTED_FIELDS:
        return None
    raise ValueError(f"no grounding rule for field: {field!r}")


def document_grounding(
    extracted: dict[str, Any],
    tokens: Sequence[str],
    fields: Iterable[str],
    *,
    decimal_sep: str | None = None,
) -> dict[str, float]:
    """Support scores for every field that has a grounding rule."""
    scores: dict[str, float] = {}
    for field in fields:
        support = grounding_support(
            field, extracted.get(field), tokens, decimal_sep=decimal_sep
        )
        if support is not None:
            scores[field] = support
    return scores

import json
from decimal import Decimal

import pytest

from underwrite.signals.grounding import (
    document_grounding,
    grounding_support,
    numeric_support,
    page_tokens,
    text_support,
)

# Real OCR tokens from a CORD receipt.
TOKENS = [
    "901016", "-TICKET", "CP", "2", "60.000", "60.000", "TOTAL", "DISC", "$",
    "-60.000", "TAX", "5.455", "Subtotal", "60.000", "TOTAL",
]

GT = (
    '{"valid_line": ['
    '{"words": [{"text": "TOTAL"}, {"text": "60.000"}]},'
    '{"words": [{"text": "TAX"}, {"text": "5.455"}]},'
    '{"words": [{"text": ""}, {"text": null}]}'
    "]}"
)


def test_page_tokens_flattens_lines_and_skips_blanks():
    assert page_tokens(GT) == ["TOTAL", "60.000", "TAX", "5.455"]


def test_page_tokens_accepts_an_already_parsed_dict():
    assert page_tokens(json.loads(GT)) == ["TOTAL", "60.000", "TAX", "5.455"]


def test_page_tokens_without_ocr_returns_empty():
    assert page_tokens("{}") == []


def test_an_exact_amount_is_grounded():
    # "60.000" on an Indonesian receipt means 60,000 rupiah.
    assert numeric_support(Decimal("60000"), TOKENS, decimal_sep="") == 1.0


def test_an_amount_that_is_not_on_the_page_is_not_grounded():
    assert numeric_support(Decimal("60"), TOKENS, decimal_sep="") == 0.0
    assert numeric_support(Decimal("999999"), TOKENS, decimal_sep="") == 0.0


def test_missing_value_is_never_grounded():
    assert numeric_support(None, TOKENS) == 0.0


def test_no_tokens_means_no_support():
    assert numeric_support(Decimal("60000"), [], decimal_sep="") == 0.0


def test_text_on_the_page_scores_high():
    assert text_support("TAX", TOKENS) > 0.9
    assert text_support("Subtotal", TOKENS) > 0.9


def test_unrelated_text_scores_low():
    assert text_support("ACME HARDWARE LIMITED", TOKENS) < 0.5


def test_missing_text_is_not_grounded():
    assert text_support(None, TOKENS) == 0.0
    assert text_support("", TOKENS) == 0.0


def test_grounding_dispatch_covers_money_text_and_skips_dates():
    assert grounding_support("total", Decimal("60000"), TOKENS, decimal_sep="") == 1.0
    assert grounding_support("vendor", "TAX", TOKENS) > 0.9
    assert grounding_support("date", "2026-10-17", TOKENS) is None


def test_document_grounding_skips_unsupported_fields():
    scores = document_grounding(
        {"total": Decimal("60000"), "vendor": "TOTAL", "date": "2026-10-17"},
        TOKENS,
        ("vendor", "date", "subtotal", "tax", "total"),
        decimal_sep="",
    )
    assert set(scores) == {"vendor", "subtotal", "tax", "total"}
    assert scores["total"] == 1.0
    # subtotal and tax were not extracted, so they are ungrounded rather than absent.
    assert scores["subtotal"] == 0.0


def test_unknown_field_has_no_rule():
    with pytest.raises(ValueError):
        grounding_support("banana", "x", TOKENS)

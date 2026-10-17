from datetime import date
from decimal import Decimal

import pytest

from underwrite.data.normalize import (
    is_correct,
    normalize_date,
    normalize_vendor,
    parse_money,
    vendor_matches,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("75,000", Decimal("75000")),
        ("43.636", Decimal("43636")),  # rupiah thousands group
        ("Rp 0.000", Decimal("0")),
        ("25.000", Decimal("25000")),
        ("₹1,200.00", Decimal("1200.00")),
        ("1,234.56", Decimal("1234.56")),
        ("1.234,56", Decimal("1234.56")),
        ("-7,800", Decimal("-7800")),
        (" 354.00 ", Decimal("354.00")),
        ("Rs. 1,23,456.78", Decimal("123456.78")),
        (".1,234.50", Decimal("1234.50")),
        ("1,234.", Decimal("1234")),
        (1000, Decimal("1000")),
    ],
)
def test_parse_money_infers_separators(raw, expected):
    assert parse_money(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "abc", "Rs.", "-"])
def test_parse_money_rejects_unreadable_values(raw):
    assert parse_money(raw) is None


def test_parse_money_with_explicit_rupiah_convention():
    assert parse_money("43.636", decimal_sep="") == Decimal("43636")
    assert parse_money("1,234", decimal_sep="") == Decimal("1234")


def test_parse_money_rejects_unknown_separator():
    with pytest.raises(ValueError):
        parse_money("1;23", decimal_sep=";")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-10-17", date(2026, 10, 17)),
        ("2026/10/17", date(2026, 10, 17)),
        ("17/10/2026", date(2026, 10, 17)),
        ("17-10-2026", date(2026, 10, 17)),
        ("17 Oct 2026", date(2026, 10, 17)),
        ("Oct 17, 2026", date(2026, 10, 17)),
    ],
)
def test_normalize_date_formats(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "not a date", "2026-13-40"])
def test_normalize_date_rejects_unreadable_values(raw):
    assert normalize_date(raw) is None


def test_normalize_vendor_folds_case_and_punctuation():
    assert normalize_vendor("  Bob's   Cafe!! ") == "bob s cafe"
    assert normalize_vendor("Café Éclair") == "cafe eclair"
    assert normalize_vendor(None) == ""


def test_vendor_matches_is_fuzzy():
    assert vendor_matches("Bob's Cafe", "Bobs Cafe")
    assert vendor_matches("SUNRISE TRADERS", "Sunrise Trader")
    assert not vendor_matches("Bob's Cafe", "Alice Hardware")


def test_vendor_ratio_is_strict_about_extra_suffixes():
    # Documented limitation of the ratio >= 90 rule: an added legal suffix drops
    # the similarity below the threshold, so the field is scored incorrect.
    assert not vendor_matches("SUNRISE TRADERS", "Sunrise Traders Pvt Ltd")


def test_is_correct_money_ignores_formatting():
    assert is_correct("total", "354.00", "354")
    assert is_correct("total", "₹1,200.00", "1200.00")
    assert not is_correct("total", "354.01", "354.00")


def test_is_correct_treats_parse_failure_as_an_error():
    assert not is_correct("total", None, "354.00")
    assert not is_correct("total", "not a number", "354.00")


def test_is_correct_when_gold_is_absent():
    assert is_correct("tax", None, None)
    assert not is_correct("tax", "5.00", None)


def test_is_correct_date_and_vendor():
    assert is_correct("date", "17/10/2026", "2026-10-17")
    assert not is_correct("date", "18/10/2026", "2026-10-17")
    assert is_correct("vendor", "Bobs Cafe", "Bob's Cafe")


def test_is_correct_rejects_unknown_field():
    with pytest.raises(ValueError):
        is_correct("banana", "a", "b")

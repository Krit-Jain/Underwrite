from datetime import date
from decimal import Decimal

import pytest

from underwrite.data.normalize import parse_money
from underwrite.data.synth_invoices import (
    MONEY_STYLES,
    _TEMPLATE_COUNT,
    build_spec,
    format_money,
    generate_specs,
    render_html,
)
from underwrite.signals.gstin import is_valid_gstin

import random


def specs(count: int = 60, seed: int = 0):
    return generate_specs(count, seed=seed)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 99])
def test_arithmetic_is_internally_consistent(seed):
    for spec in specs(seed=seed):
        assert spec.subtotal == sum((item.amount for item in spec.items), Decimal("0"))
        assert spec.total == spec.subtotal + spec.tax
        if spec.inter_state:
            assert spec.igst == spec.tax
            assert spec.cgst == spec.sgst == Decimal("0.00")
        else:
            assert spec.cgst + spec.sgst == spec.tax
            assert spec.igst == Decimal("0.00")


def test_every_amount_is_quantized_to_two_places():
    for spec in specs():
        for amount in (spec.subtotal, spec.tax, spec.cgst, spec.sgst, spec.igst, spec.total):
            assert amount == amount.quantize(Decimal("0.01"))
        for item in spec.items:
            assert item.amount == item.amount.quantize(Decimal("0.01"))


def test_vendor_gstin_is_valid_and_checksum_correct():
    for spec in specs():
        assert is_valid_gstin(spec.vendor_gstin, require_known_state=True)
        if spec.buyer_gstin is not None:
            assert is_valid_gstin(spec.buyer_gstin, require_known_state=True)


@pytest.mark.parametrize("style", MONEY_STYLES)
def test_rendered_money_round_trips_through_the_parser(style):
    for spec in specs(20):
        for amount in (spec.subtotal, spec.tax, spec.total, spec.items[0].unit_price):
            assert parse_money(format_money(amount, style)) == amount


def test_date_is_rendered_in_the_configured_format():
    for spec in specs(20):
        assert spec.date.strftime(spec.date_format)


def test_gold_matches_the_specification():
    for spec in specs(20):
        gold = spec.gold
        assert parse_money(gold["total"]) == spec.total
        assert parse_money(gold["subtotal"]) == spec.subtotal
        assert parse_money(gold["tax"]) == spec.tax
        assert gold["date"] == spec.date.isoformat()
        assert gold["vendor"] == spec.vendor_name


def test_generation_is_deterministic():
    assert generate_specs(5, seed=3) == generate_specs(5, seed=3)
    assert generate_specs(5, seed=3) != generate_specs(5, seed=4)


def test_templates_and_formatting_vary():
    generated = specs(120)
    assert {spec.template for spec in generated} == set(range(_TEMPLATE_COUNT))
    assert len({spec.money_style for spec in generated}) > 1
    assert len({spec.date_format for spec in generated}) > 1
    assert any(spec.inter_state for spec in generated)
    assert any(not spec.inter_state for spec in generated)


def test_html_shows_the_displayed_total_and_gstin():
    for spec in specs(30):
        html = render_html(spec)
        assert format_money(spec.total, spec.money_style) in html
        assert spec.vendor_gstin in html
        assert spec.invoice_no in html


def test_html_shows_the_correct_tax_lines():
    for spec in specs(30):
        html = render_html(spec)
        if spec.inter_state:
            assert "IGST" in html
        else:
            assert "CGST" in html and "SGST" in html


@pytest.mark.parametrize(
    "amount,style,expected",
    [
        (Decimal("123456.78"), "grouped", "1,23,456.78"),
        (Decimal("1000"), "grouped", "1,000.00"),
        (Decimal("10000000"), "grouped", "1,00,00,000.00"),
        (Decimal("999.5"), "plain", "999.50"),
        (Decimal("1234.5"), "symbol", "\u20b91,234.50"),
        (Decimal("-2500"), "grouped", "-2,500.00"),
    ],
)
def test_indian_money_formatting(amount, style, expected):
    assert format_money(amount, style) == expected

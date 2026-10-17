import pytest

from underwrite.signals.verbalized import parse_verbalized, verbalized_instruction

FIELDS = ("vendor", "date", "subtotal", "tax", "total")


def test_instruction_names_every_field():
    instruction = verbalized_instruction(FIELDS)
    for name in FIELDS:
        assert name in instruction


def test_parses_a_plain_object():
    text = '{"total": 0.9, "tax": 0.4}'
    assert parse_verbalized(text, FIELDS) == {"total": 0.9, "tax": 0.4}


def test_parses_a_fenced_block():
    text = '```json\n{"total": 0.75}\n```'
    assert parse_verbalized(text, FIELDS) == {"total": 0.75}


def test_clamps_values_into_the_unit_interval():
    text = '{"total": 1.8, "tax": -0.5}'
    assert parse_verbalized(text, FIELDS) == {"total": 1.0, "tax": 0.0}


def test_ignores_booleans_and_strings():
    text = '{"total": true, "tax": "0.5", "subtotal": 0.3}'
    assert parse_verbalized(text, FIELDS) == {"subtotal": 0.3}


def test_ignores_fields_that_were_not_requested():
    text = '{"total": 0.8, "banana": 0.2}'
    assert parse_verbalized(text, FIELDS) == {"total": 0.8}


def test_returns_nothing_for_unparseable_output():
    assert parse_verbalized("I am fairly confident!", FIELDS) == {}


def test_accepts_integers():
    assert parse_verbalized('{"total": 1, "tax": 0}', FIELDS) == {"total": 1.0, "tax": 0.0}

from datetime import date
from decimal import Decimal

import pytest

from underwrite.signals.validators import CHECK_NAMES, as_vector, validate_document

TODAY = date(2026, 10, 17)


def indian_doc(**overrides):
    fields = {
        "vendor": "SUNRISE TRADERS",
        "date": "17/10/2026",
        "subtotal": "8,301.50",
        "tax": "996.18",
        "total": "9,297.68",
    }
    fields.update(overrides)
    return fields


def test_a_consistent_indian_invoice_passes_every_check():
    checks = validate_document(indian_doc(), today=TODAY)
    assert checks == {
        "valid_money_format": 1.0,
        "valid_date_format": 1.0,
        "plausible_date": 1.0,
        "sum_matches": 1.0,
    }


def test_a_consistent_cord_receipt_passes_with_inference():
    fields = {"subtotal": "43.636", "tax": "4.364", "total": "48.000"}
    assert validate_document(fields, today=TODAY)["sum_matches"] == 1.0


def test_a_consistent_cord_receipt_passes_with_pinned_convention():
    fields = {"subtotal": "43.636", "tax": "4.364", "total": "48.000"}
    assert validate_document(fields, today=TODAY, decimal_sep="")["sum_matches"] == 1.0


def test_arithmetic_mismatch_is_caught():
    checks = validate_document(indian_doc(total="9,297.99"), today=TODAY)
    assert checks["sum_matches"] == 0.0


def test_difference_of_exactly_the_tolerance_still_passes():
    checks = validate_document(indian_doc(total="9,297.69"), today=TODAY)
    assert checks["sum_matches"] == 1.0
    over = validate_document(indian_doc(total="9,297.70"), today=TODAY)
    assert over["sum_matches"] == 0.0


def test_unparseable_money_is_flagged_but_other_checks_still_run():
    checks = validate_document(indian_doc(total="N/A"), today=TODAY)
    assert checks["valid_money_format"] == 0.0
    assert "sum_matches" not in checks
    assert checks["plausible_date"] == 1.0


def test_absent_fields_produce_no_check_rather_than_a_failure():
    checks = validate_document({"total": "100.00"}, today=TODAY)
    assert checks == {"valid_money_format": 1.0}


def test_empty_fields_produce_no_checks():
    assert validate_document({}, today=TODAY) == {}


@pytest.mark.parametrize("value", ["", "not a date", "2026-13-40"])
def test_unparseable_date_is_flagged(value):
    checks = validate_document({"total": "1.00", "date": value}, today=TODAY)
    assert checks["valid_date_format"] == 0.0
    assert "plausible_date" not in checks


def test_a_null_date_is_absent_not_invalid():
    checks = validate_document({"total": "1.00", "date": None}, today=TODAY)
    assert "valid_date_format" not in checks
    assert "plausible_date" not in checks


def test_future_date_beyond_the_slack_is_implausible():
    checks = validate_document({"date": "2027-01-01"}, today=TODAY)
    assert checks["plausible_date"] == 0.0


def test_todays_date_is_plausible():
    assert validate_document({"date": "2026-10-17"}, today=TODAY)["plausible_date"] == 1.0


def test_ancient_date_is_implausible():
    checks = validate_document({"date": "1995-01-01"}, today=TODAY)
    assert checks["plausible_date"] == 0.0


def test_gstin_check_only_runs_when_a_gstin_is_present():
    assert "valid_gstin" not in validate_document(indian_doc(), today=TODAY)
    assert validate_document({"gstin": "27AAACR5055K1Z7"}, today=TODAY)["valid_gstin"] == 1.0
    assert validate_document({"gstin": "27AAACR5055K1Z0"}, today=TODAY)["valid_gstin"] == 0.0


def test_never_issued_state_code_is_rejected_as_a_gstin():
    assert validate_document({"gstin": "00AAACR5055K1ZN"}, today=TODAY)["valid_gstin"] == 0.0


def test_tax_split_is_checked_when_both_halves_are_present():
    good = {"cgst": "498.09", "sgst": "498.09", "tax": "996.18"}
    assert validate_document(good, today=TODAY)["tax_split_matches"] == 1.0
    bad = {"cgst": "498.09", "sgst": "400.00", "tax": "996.18"}
    assert validate_document(bad, today=TODAY)["tax_split_matches"] == 0.0


def test_tax_split_is_omitted_when_a_half_is_missing():
    assert "tax_split_matches" not in validate_document({"tax": "996.18"}, today=TODAY)


def test_as_vector_fills_absent_checks_and_keeps_the_canonical_order():
    vector = as_vector({"sum_matches": 1.0})
    assert list(vector) == list(CHECK_NAMES)
    assert vector["sum_matches"] == 1.0
    assert vector["valid_gstin"] == 0.0
    assert all(isinstance(value, float) for value in vector.values())

import pytest

from underwrite.app.certificate import (
    DECISION_AUTO,
    DECISION_REVIEW,
    build_certificate,
)


def make(decisions, **overrides):
    params = {
        "alpha": 0.10,
        "delta": 0.1,
        "threshold": 0.8,
        "model_id": "google/gemma-4-E2B-it",
        "calibration_documents": 289,
        "audit_sample": 3,
    }
    params.update(overrides)
    return build_certificate(decisions, **params)


def test_counts_and_rates_come_from_the_decisions():
    certificate = make([DECISION_AUTO] * 30 + [DECISION_REVIEW] * 10)
    assert certificate.total_fields == 40
    assert certificate.auto_fields == 30
    assert certificate.review_fields == 10
    assert certificate.auto_rate == pytest.approx(0.75)


def test_an_empty_batch_does_not_divide_by_zero():
    certificate = make([])
    assert certificate.total_fields == 0
    assert certificate.auto_rate == 0.0


def test_unknown_decisions_are_rejected():
    with pytest.raises(ValueError):
        make(["auto", "maybe"])


def test_an_uncertified_batch_says_so_and_makes_no_error_promise():
    certificate = make([DECISION_REVIEW] * 5, threshold=None, audit_sample=0)
    assert not certificate.certified
    caveat = certificate.caveat()
    assert "not certified" in caveat
    assert "at most" not in caveat


def test_the_caveat_states_the_guarantee_and_the_assumption():
    caveat = make([DECISION_AUTO] * 8 + [DECISION_REVIEW] * 2).caveat()
    assert "90%" in caveat, "confidence should be 1 - delta"
    assert "10%" in caveat, "budget should appear"
    assert "289" in caveat, "calibration size should appear"
    assert "resemble the calibration ones" in caveat
    assert "silently" in caveat


def test_the_caveat_promises_a_long_run_rate_not_a_property_of_this_batch():
    caveat = make([DECISION_AUTO] * 8 + [DECISION_REVIEW] * 2).caveat()
    assert "long-run error rate" in caveat
    assert "not the guarantee" in caveat
    # The old wording claimed the rate was bounded among this batch's fields.
    assert "among the" not in caveat


def test_caveat_numbers_match_the_record():
    certificate = make([DECISION_AUTO] * 3 + [DECISION_REVIEW] * 1, audit_sample=2)
    caveat = certificate.caveat()
    assert f"{certificate.calibration_documents}" in caveat
    assert f"{certificate.auto_rate:.0%}" in caveat
    assert f"{certificate.audit_sample} auto-accepted fields" in caveat


def test_serialisation_round_trips_the_fields_the_ui_needs():
    payload = make([DECISION_AUTO, DECISION_REVIEW]).as_dict()
    for key in ("certified", "alpha", "delta", "threshold", "auto_rate", "caveat", "counts"):
        assert key in payload
    assert payload["counts"] == {"auto": 1, "review": 1}

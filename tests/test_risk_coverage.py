import math

import numpy as np
import pytest

from underwrite.eval.risk_coverage import (
    aurc,
    auroc,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
    summarise,
)

SCORES = [0.9, 0.8, 0.1, 0.0]  # descending already
ERRORS = [0, 0, 1, 1]


def test_curve_covers_the_whole_set_with_uniform_steps():
    curve = risk_coverage_curve(SCORES, ERRORS)
    assert np.allclose(curve.coverage, [0.25, 0.5, 0.75, 1.0])


def test_risk_is_the_error_rate_among_accepted_items():
    curve = risk_coverage_curve(SCORES, ERRORS)
    assert np.allclose(curve.risk, [0.0, 0.0, 1 / 3, 0.5])


def test_risk_at_full_coverage_is_the_base_error_rate():
    curve = risk_coverage_curve(SCORES, ERRORS)
    assert curve.risk[-1] == pytest.approx(np.mean(ERRORS))


def test_curve_orders_by_descending_score():
    # Same data, shuffled input: the ranking (not the input order) decides the curve.
    curve = risk_coverage_curve([0.1, 0.9, 0.0, 0.8], [1, 0, 1, 0])
    assert np.allclose(curve.risk, [0.0, 0.0, 1 / 3, 0.5])


def test_auroc_is_one_for_a_perfect_ranking():
    assert auroc(SCORES, ERRORS) == pytest.approx(1.0)


def test_auroc_is_half_for_a_constant_score():
    assert auroc([0.5] * 4, ERRORS) == pytest.approx(0.5)


def test_auroc_is_nan_with_a_single_class():
    assert math.isnan(auroc(SCORES, [0, 0, 0, 0]))


def test_aurc_is_the_mean_of_the_risk_curve():
    expected = np.mean([0.0, 0.0, 1 / 3, 0.5])
    assert aurc(SCORES, ERRORS) == pytest.approx(expected)


def test_a_better_ranking_has_a_lower_aurc():
    good = aurc([0.9, 0.8, 0.1, 0.0], ERRORS)
    bad = aurc([0.0, 0.1, 0.8, 0.9], ERRORS)
    assert good < bad


def test_risk_at_coverage_reads_the_curve():
    assert risk_at_coverage(SCORES, ERRORS, 0.5) == pytest.approx(0.0)
    assert risk_at_coverage(SCORES, ERRORS, 1.0) == pytest.approx(0.5)


def test_risk_at_coverage_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        risk_at_coverage(SCORES, ERRORS, 0.0)
    with pytest.raises(ValueError):
        risk_at_coverage(SCORES, ERRORS, 1.5)


def test_coverage_at_risk_finds_the_largest_acceptable_prefix():
    assert coverage_at_risk(SCORES, ERRORS, 0.05) == pytest.approx(0.5)
    assert coverage_at_risk(SCORES, ERRORS, 0.0) == pytest.approx(0.5)


def test_coverage_at_risk_is_zero_when_nothing_qualifies():
    assert coverage_at_risk([0.5, 0.4], [1, 1], 0.0) == pytest.approx(0.0)


def test_summarise_reports_every_target_risk():
    summary = summarise(SCORES, ERRORS)
    assert set(summary) == {"auroc", "aurc", "coverage@0.02", "coverage@0.05", "coverage@0.10"}


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        risk_coverage_curve([0.1, 0.2], [1])


def test_an_empty_curve_is_rejected():
    with pytest.raises(ValueError):
        risk_coverage_curve([], [])

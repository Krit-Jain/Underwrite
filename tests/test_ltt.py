import numpy as np
import pytest

from underwrite.conformal.ltt import (
    certify,
    certify_threshold,
    default_grid,
    minimum_accepts_for_zero_errors,
    selective_risk,
)


def test_default_grid_is_fixed_and_covers_the_unit_interval():
    grid = default_grid()
    assert len(grid) == 501
    assert grid[0] == 0.0 and grid[-1] == 1.0


def test_minimum_accepts_matches_the_sample_size_reality():
    assert minimum_accepts_for_zero_errors(0.05, 0.1) == 45
    assert minimum_accepts_for_zero_errors(0.02, 0.1) == 114
    assert minimum_accepts_for_zero_errors(0.01, 0.1) == 230


def test_minimum_accepts_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        minimum_accepts_for_zero_errors(0.0, 0.1)
    with pytest.raises(ValueError):
        minimum_accepts_for_zero_errors(0.05, 1.0)


def test_error_free_calibration_certifies():
    scores = np.linspace(0, 1, 100)
    errors = np.zeros(100, dtype=int)
    assert certify_threshold(scores, errors, 0.05, 0.1, default_grid()) is not None


def test_too_few_items_cannot_certify():
    scores = np.linspace(0, 1, 10)
    errors = np.zeros(10, dtype=int)
    assert certify_threshold(scores, errors, 0.05, 0.1, default_grid()) is None


def test_all_wrong_calibration_cannot_certify():
    scores = np.linspace(0, 1, 500)
    errors = np.ones(500, dtype=int)
    assert certify_threshold(scores, errors, 0.05, 0.1, default_grid()) is None


def test_fixed_sequence_stops_at_the_first_failure():
    """After a powered threshold fails, later (lenient) thresholds are never tested.

    That stop is what controls the family-wise error rate. Here the strictest
    *powered* threshold carries a 4% error rate, which still fails at alpha=5% with
    only 50 items -- so a lenient threshold that would have passed must not be used.
    """
    strict_scores = np.full(50, 1.0)
    strict_errors = np.array([1, 1] + [0] * 48, dtype=int)
    bulk_scores = np.full(350, 0.5)
    bulk_errors = np.zeros(350, dtype=int)

    scores = np.concatenate([strict_scores, bulk_scores])
    errors = np.concatenate([strict_errors, bulk_errors])

    assert certify_threshold(scores, errors, 0.05, 0.1, default_grid()) is None


def test_thresholds_without_power_are_skipped():
    """A strict threshold accepting a handful of items must not halt certification."""
    scores = np.linspace(0, 1, 100)
    errors = np.zeros(100, dtype=int)
    # The strictest non-empty threshold accepts one item and cannot reject.
    assert certify_threshold(scores, errors, 0.05, 0.1, default_grid()) is not None


def test_grid_order_does_not_matter():
    scores = np.linspace(0, 1, 200)
    errors = (scores < 0.3).astype(int)
    ordered = certify_threshold(scores, errors, 0.05, 0.1, default_grid())
    shuffled = certify_threshold(scores, errors, 0.05, 0.1, np.random.default_rng(0).permutation(default_grid()))
    assert ordered == shuffled


def test_certification_reports_the_accepted_prefix():
    scores = np.linspace(0, 1, 100)
    errors = (scores < 0.5).astype(int)
    result = certify(scores, errors, 0.10, 0.1)
    assert result.certified
    assert result.n_accepted > 0
    assert result.coverage == pytest.approx(result.n_accepted / 100)
    assert result.selective_risk <= 0.10


def test_uncertified_result_reports_zero_coverage():
    result = certify([0.5, 0.5], [1, 1], 0.05, 0.1)
    assert not result.certified
    assert result.threshold is None
    assert result.coverage == 0.0


def test_selective_risk_of_an_empty_prefix_is_nan():
    assert np.isnan(selective_risk([0.1, 0.2], [1, 1], 0.9))


def _draw(rng, size):
    """Scores with a clean top: the highest-scoring items are almost always correct."""
    scores = rng.random(size)
    probability_error = 0.5 * (1.0 - scores) ** 2
    errors = (rng.random(size) < probability_error).astype(int)
    return scores, errors


def test_realised_risk_exceeds_alpha_at_most_delta_of_the_time():
    """The guarantee, simulated.

    The certified claim is about the *population* selective risk, so the test draw is
    deliberately large: with a small test set, the realised risk fluctuates around
    alpha and exceeds it far more often than delta even when the procedure is sound.
    """
    rng = np.random.default_rng(0)
    alpha, delta, trials = 0.05, 0.1, 200
    grid = np.linspace(0.0, 1.0, 101)

    violations = 0
    certified = 0
    coverages = []
    for _ in range(trials):
        cal_scores, cal_errors = _draw(rng, 400)
        test_scores, test_errors = _draw(rng, 20_000)

        result = certify(cal_scores, cal_errors, alpha, delta, grid)
        if not result.certified:
            continue
        certified += 1
        coverages.append(float((test_scores >= result.threshold).mean()))
        if selective_risk(test_scores, test_errors, result.threshold) > alpha:
            violations += 1

    assert certified > trials * 0.5, "the gate should usually certify on this data"
    assert violations / trials <= delta + 0.03
    assert np.mean(coverages) > 0.05

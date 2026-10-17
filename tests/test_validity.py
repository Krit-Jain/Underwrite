import numpy as np
import pandas as pd
import pytest

from underwrite.eval.validity import evaluate_methods, resplit_validity


def draw(rng, size):
    scores = rng.random(size)
    probability_error = 0.5 * (1.0 - scores) ** 2
    errors = (rng.random(size) < probability_error).astype(int)
    return scores, errors


def test_a_useful_score_certifies_and_respects_the_budget():
    rng = np.random.default_rng(0)
    scores, errors = draw(rng, 2000)
    summary = resplit_validity(scores, errors, 0.05, 0.1, n_splits=100, seed=1)

    assert summary.certified_rate > 0.5
    assert summary.violation_rate <= 0.2
    assert summary.mean_coverage > 0.05
    assert summary.mean_realised_risk <= 0.10


def test_an_uninformative_score_cannot_certify_often():
    rng = np.random.default_rng(2)
    scores = rng.random(1000)
    errors = (rng.random(1000) < 0.5).astype(int)
    summary = resplit_validity(scores, errors, 0.05, 0.1, n_splits=50, seed=3)

    # With no signal the top of the ranking offers no protection.
    assert summary.mean_coverage < 0.5


def test_resplits_are_deterministic_for_a_seed():
    rng = np.random.default_rng(0)
    scores, errors = draw(rng, 500)
    first = resplit_validity(scores, errors, 0.10, 0.1, n_splits=25, seed=7)
    second = resplit_validity(scores, errors, 0.10, 0.1, n_splits=25, seed=7)
    assert first == second


def test_different_seeds_give_different_splits():
    rng = np.random.default_rng(0)
    scores, errors = draw(rng, 500)
    first = resplit_validity(scores, errors, 0.10, 0.1, n_splits=25, seed=1)
    second = resplit_validity(scores, errors, 0.10, 0.1, n_splits=25, seed=2)
    assert first != second


def test_summary_reports_the_pieces_it_needs():
    rng = np.random.default_rng(0)
    scores, errors = draw(rng, 500)
    summary = resplit_validity(scores, errors, 0.05, 0.1, n_splits=20, seed=0)
    assert summary.n_splits == 20
    assert 0.0 <= summary.certified_rate <= 1.0
    assert 0.0 <= summary.violation_rate <= 1.0
    assert summary.p95_realised_risk >= summary.mean_realised_risk - 1e-9


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        resplit_validity([0.1, 0.2, 0.3], [1, 0], 0.05, 0.1)


def test_too_few_items_are_rejected():
    with pytest.raises(ValueError):
        resplit_validity([0.1, 0.2], [1, 0], 0.05, 0.1)


def test_invalid_calibration_fraction_is_rejected():
    with pytest.raises(ValueError):
        resplit_validity([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], 0.05, 0.1, cal_fraction=1.0)


def test_evaluate_methods_covers_every_method_and_alpha():
    rng = np.random.default_rng(0)
    scores, errors = draw(rng, 600)
    scored = pd.DataFrame(
        {
            "doc_id": [f"doc{i}" for i in range(600)],
            "split": ["calibration"] * 300 + ["test"] * 300,
            "correct": 1 - errors,
            "strong": scores,
            "weak": rng.random(600),
        }
    )
    table = evaluate_methods(scored, ["strong", "weak"], [0.05, 0.10], 0.1, n_splits=10, seed=0)
    assert len(table) == 4
    assert set(table["method"]) == {"strong", "weak"}
    assert set(table["alpha"]) == {0.05, 0.10}

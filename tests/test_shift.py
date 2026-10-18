import numpy as np
import pandas as pd
import pytest

from underwrite.eval.shift import calibrate_and_deploy, repair_curve, shift_table


def draw(rng, size, error_scale=0.5):
    scores = rng.random(size)
    probability_error = error_scale * (1.0 - scores) ** 2
    errors = (rng.random(size) < probability_error).astype(int)
    return scores, errors


def test_deploying_on_the_same_distribution_keeps_the_promise():
    rng = np.random.default_rng(0)
    source_scores, source_errors = draw(rng, 1500)
    target_scores, target_errors = draw(rng, 800)

    row = calibrate_and_deploy(
        source_scores, source_errors, target_scores, target_errors, 0.10, 0.1
    )
    assert row["certified"] is True
    assert row["target_risk"] <= 0.15
    assert row["violated"] is False


def test_a_shifted_corpus_breaks_the_promise():
    rng = np.random.default_rng(1)
    source_scores, source_errors = draw(rng, 1500, error_scale=0.5)
    # The target looks the same to the score, but its labels are far worse.
    target_scores, _ = draw(rng, 800, error_scale=0.5)
    target_errors = (rng.random(800) < 0.6).astype(int)

    row = calibrate_and_deploy(
        source_scores, source_errors, target_scores, target_errors, 0.10, 0.1
    )
    assert row["certified"] is True
    assert row["target_risk"] > 0.10
    assert row["violated"] is True


def test_an_uncertifiable_source_reports_no_target_result():
    row = calibrate_and_deploy([0.5, 0.5], [1, 1], [0.5], [1], 0.05, 0.1)
    assert row["certified"] is False
    assert row["threshold"] is None
    assert row["target_coverage"] == 0.0
    assert row["violated"] is False


def test_shift_table_covers_every_method_and_alpha():
    rng = np.random.default_rng(2)
    scores, errors = draw(rng, 500)
    frame = pd.DataFrame(
        {
            "correct": 1 - errors,
            "strong": scores,
            "weak": rng.random(500),
        }
    )
    table = shift_table(frame, frame, ["strong", "weak"], [0.05, 0.10], 0.1)
    assert len(table) == 4
    assert set(table["method"]) == {"strong", "weak"}
    assert set(table["alpha"]) == {0.05, 0.10}


def test_repair_curve_reports_every_k():
    rng = np.random.default_rng(3)
    source_scores, source_errors = draw(rng, 800)
    target_scores, target_errors = draw(rng, 300)

    table = repair_curve(
        source_scores, source_errors, target_scores, target_errors,
        ks=[10, 50], alpha=0.10, delta=0.1, n_draws=10, seed=4,
    )
    assert list(table["k"]) == [10, 50]
    assert (table["certified_rate"] <= 1.0).all()


def test_more_labelled_target_documents_do_not_hurt_certification():
    rng = np.random.default_rng(5)
    source_scores, source_errors = draw(rng, 800, error_scale=0.5)
    target_scores, target_errors = draw(rng, 400, error_scale=0.9)

    table = repair_curve(
        source_scores, source_errors, target_scores, target_errors,
        ks=[10, 150], alpha=0.10, delta=0.1, n_draws=25, seed=6,
    )
    small, large = table.iloc[0], table.iloc[1]
    assert large["certified_rate"] >= small["certified_rate"]


def test_repair_curve_is_deterministic():
    rng = np.random.default_rng(7)
    source_scores, source_errors = draw(rng, 400)
    target_scores, target_errors = draw(rng, 200)
    first = repair_curve(source_scores, source_errors, target_scores, target_errors,
                         ks=[20], alpha=0.1, delta=0.1, n_draws=5, seed=8)
    second = repair_curve(source_scores, source_errors, target_scores, target_errors,
                          ks=[20], alpha=0.1, delta=0.1, n_draws=5, seed=8)
    assert first.equals(second)


def test_repair_curve_rejects_k_larger_than_the_corpus():
    with pytest.raises(ValueError):
        repair_curve([0.1, 0.2], [0, 0], [0.1, 0.2], [0, 0], ks=[5], alpha=0.1, delta=0.1)


def test_repair_curve_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        repair_curve([0.1, 0.2], [0], [0.1, 0.2], [0, 0], ks=[1], alpha=0.1, delta=0.1)


def test_target_only_repair_ignores_the_source_corpus():
    """Calibrating on the target documents alone is what a new client template needs."""
    rng = np.random.default_rng(11)
    target_scores, target_errors = draw(rng, 400, error_scale=0.5)
    # A source corpus that would certify everything: it must not leak into the repair.
    source_scores = np.linspace(0, 1, 400)
    source_errors = np.zeros(400, dtype=int)

    pooled = repair_curve(
        source_scores, source_errors, target_scores, target_errors,
        ks=[100], alpha=0.10, delta=0.1, n_draws=5, seed=12, include_source=True,
    )
    target_only = repair_curve(
        source_scores, source_errors, target_scores, target_errors,
        ks=[100], alpha=0.10, delta=0.1, n_draws=5, seed=12, include_source=False,
    )
    assert not pooled.equals(target_only)

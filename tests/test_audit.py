import numpy as np
import pytest

from underwrite.conformal.audit import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    injected_drift,
    per_look_confidence,
    simulate_audit,
)

ALPHA = 0.10


def test_upper_bound_never_exceeds_one():
    assert clopper_pearson_upper(0, 1) <= 1.0
    assert clopper_pearson_upper(5, 5) == 1.0


def test_upper_bound_grows_with_observed_errors():
    assert clopper_pearson_upper(5, 100) > clopper_pearson_upper(1, 100)


def test_lower_bound_is_zero_without_errors():
    assert clopper_pearson_lower(0, 100) == 0.0


def test_lower_bound_grows_with_more_errors():
    bounds = [clopper_pearson_lower(k, 100) for k in (1, 5, 20, 50)]
    assert bounds == sorted(bounds)


def test_lower_bound_sits_below_the_observed_rate():
    assert clopper_pearson_lower(40, 100) < 0.4


def test_lower_bound_rejects_impossible_counts():
    with pytest.raises(ValueError):
        clopper_pearson_lower(5, 2)


def test_upper_bound_brackets_the_lower_one():
    assert clopper_pearson_lower(10, 100) < clopper_pearson_upper(10, 100)


def test_per_look_confidence_spends_the_budget():
    assert per_look_confidence(0.05, 1) == pytest.approx(0.95)
    assert per_look_confidence(0.05, 10) == pytest.approx(0.995)


def test_per_look_confidence_validates_its_inputs():
    with pytest.raises(ValueError):
        per_look_confidence(0.0, 5)
    with pytest.raises(ValueError):
        per_look_confidence(0.05, 0)


def test_audit_rate_must_be_a_fraction():
    with pytest.raises(ValueError):
        simulate_audit([0, 1, 0], audit_rate=0.0)


def test_a_clean_stream_never_alarms():
    result = simulate_audit([0] * 2000, audit_rate=0.2, alpha=ALPHA, seed=1)
    assert result.first_alarm(ALPHA) is None
    assert max(result.lower_bounds) == 0.0


def test_a_badly_broken_stream_alarms():
    result = simulate_audit([1] * 500, audit_rate=1.0, alpha=ALPHA, check_every=5, seed=0)
    assert result.first_alarm(ALPHA) is not None


def test_drift_is_detected_only_after_the_switch():
    stream = injected_drift(1000, 1000, clean_error=0.01, drift_error=0.5, seed=3)
    result = simulate_audit(stream, audit_rate=0.3, alpha=ALPHA, check_every=5, seed=4)
    alarm = result.first_alarm(ALPHA)
    assert alarm is not None
    assert alarm > 1000, "the alarm must not fire while the stream is still clean"


def test_auditing_everything_observes_every_error():
    result = simulate_audit([0, 1, 0, 1], audit_rate=1.0, alpha=0.5, check_every=1, seed=0)
    assert result.audited[-1] == 4
    assert result.errors[-1] == 2


def test_result_frame_has_one_row_per_check():
    result = simulate_audit([0] * 100, audit_rate=1.0, check_every=10, seed=0)
    frame = result.as_frame()
    assert len(frame) == 10
    assert list(frame.columns) == ["position", "audited", "errors", "lower_bound", "upper_bound"]


def _stream(length, error_rate, seed):
    rng = np.random.default_rng(seed)
    return (rng.random(length) < error_rate).astype(int)


# The audit selection must use a seed independent of the stream's, or it replays the
# same random draws and the sample becomes correlated with the errors.
AUDIT_SEED_OFFSET = 1_000_003


def test_audit_seed_must_be_independent_of_the_stream_seed():
    """Reusing the seed correlates the audit sample with the errors."""
    stream = _stream(600, 0.06, seed=0)
    same_seed = simulate_audit(stream, audit_rate=0.1, alpha=ALPHA, check_every=10, seed=0)
    independent = simulate_audit(
        stream, audit_rate=0.1, alpha=ALPHA, check_every=10, seed=AUDIT_SEED_OFFSET
    )
    assert same_seed.errors[-1] / same_seed.audited[-1] >= 0.4, "the footgun is real"
    assert independent.errors[-1] / independent.audited[-1] < 0.2


def test_false_alarm_rate_stays_within_the_budget_on_clean_streams():
    """500 clean streams whose true error is below alpha must rarely alarm."""
    alarms = 0
    seeds = range(500)
    for seed in seeds:
        stream = _stream(600, 0.06, seed)
        result = simulate_audit(
            stream, audit_rate=0.1, alpha=ALPHA, check_every=10, seed=seed + AUDIT_SEED_OFFSET
        )
        alarms += result.first_alarm(ALPHA) is not None
    rate = alarms / len(seeds)
    assert rate <= 0.05, f"false-alarm rate {rate:.3f} exceeded the 5% budget"


def test_detection_delay_is_finite_and_reported():
    """A 40% drift is caught in most streams, a few hundred documents later."""
    delays = []
    for seed in range(200):
        stream = injected_drift(500, 500, clean_error=0.01, drift_error=0.4, seed=seed)
        result = simulate_audit(
            stream, audit_rate=0.2, alpha=ALPHA, check_every=5, seed=seed + AUDIT_SEED_OFFSET
        )
        alarm = result.first_alarm(ALPHA)
        if alarm is not None:
            delays.append(alarm - 500)
    assert len(delays) >= 120, f"only {len(delays)}/200 drifts detected"
    assert np.median(delays) < 500, "detection should arrive inside the drifted segment"

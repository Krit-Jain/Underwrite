import numpy as np
import pytest

from underwrite.conformal.audit import (
    clopper_pearson_upper,
    injected_drift,
    simulate_audit,
)


def test_upper_bound_never_exceeds_one():
    assert clopper_pearson_upper(0, 1) <= 1.0
    assert clopper_pearson_upper(5, 5) == 1.0


def test_upper_bound_is_one_for_an_empty_sample():
    assert clopper_pearson_upper(0, 0) == 1.0


def test_upper_bound_shrinks_with_more_error_free_audits():
    bounds = [clopper_pearson_upper(0, n) for n in (10, 50, 200, 1000)]
    assert bounds == sorted(bounds, reverse=True)


def test_upper_bound_grows_with_observed_errors():
    assert clopper_pearson_upper(5, 100) > clopper_pearson_upper(1, 100)
    assert clopper_pearson_upper(20, 100) > clopper_pearson_upper(5, 100)


def test_zero_errors_in_a_hundred_audits_is_about_two_percent():
    # 1 - 0.1 ** (1/100) ~= 0.0228
    assert clopper_pearson_upper(0, 100, 0.90) == pytest.approx(0.0228, abs=1e-3)


def test_audit_rejects_invalid_counts():
    with pytest.raises(ValueError):
        clopper_pearson_upper(5, 2)


def test_audit_rate_must_be_a_fraction():
    with pytest.raises(ValueError):
        simulate_audit([0, 1, 0], audit_rate=0.0)


def test_a_clean_stream_never_alarms():
    result = simulate_audit([0] * 2000, audit_rate=0.2, alpha=0.10, seed=1)
    assert result.first_alarm(0.10) is None
    assert result.upper_bounds[-1] < 0.10


def test_a_badly_broken_stream_alarms():
    result = simulate_audit([1] * 500, audit_rate=1.0, alpha=0.10, check_every=5, seed=0)
    assert result.first_alarm(0.10) is not None


def test_drift_is_detected_only_after_the_switch():
    stream = injected_drift(clean_length=1000, drift_length=1000, clean_error=0.0, drift_error=0.5, seed=3)
    result = simulate_audit(stream, audit_rate=0.3, alpha=0.10, check_every=5, seed=4)
    alarm = result.first_alarm(0.10)
    assert alarm is not None
    assert alarm > 900, "the alarm must not fire while the stream is still clean"


def test_auditing_everything_observes_every_error():
    result = simulate_audit([0, 1, 0, 1], audit_rate=1.0, alpha=0.5, check_every=1, seed=0)
    assert result.audited[-1] == 4
    assert result.errors[-1] == 2


def test_result_frame_has_one_row_per_check():
    result = simulate_audit([0] * 100, audit_rate=1.0, check_every=10, minimum_audits=1, seed=0)
    frame = result.as_frame()
    assert len(frame) == 10
    assert list(frame.columns) == ["position", "audited", "errors", "upper_bound"]


def test_checks_only_start_once_the_bound_could_support_alpha():
    result = simulate_audit([0] * 100, audit_rate=1.0, check_every=10, alpha=0.10, seed=0)
    assert result.audited[0] >= 22, "a bound from a tiny sample cannot support alpha"

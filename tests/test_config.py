from pathlib import Path

import pytest

from underwrite.config import ConfigError, config_from_mapping, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def base_config(**overrides):
    data = {
        "seed": 7,
        "model": {"id": "google/gemma-3-4b-it"},
        "splits": {"probe_train": 0.4, "calibration": 0.3, "test": 0.3},
        "conformal": {"alpha_grid": [0.10, 0.05, 0.02], "delta": 0.1},
    }
    data.update(overrides)
    return data


def test_defaults_are_applied():
    config = config_from_mapping(base_config())
    assert config.model.quantization == "nf4"
    assert config.model.compute_dtype == "bfloat16"
    assert config.model.do_sample is False
    assert config.conformal.threshold_grid_size == 501
    assert config.data.splits_file == "data/splits.json"
    assert config.signals.fields == ("vendor", "date", "subtotal", "tax", "total")


def test_repository_default_config_is_valid():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert len(config.conformal.alpha_grid) == 3
    assert abs(sum(
        [config.splits.probe_train, config.splits.calibration, config.splits.test]
    ) - 1.0) < 1e-9


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ConfigError, match="sum to 1.0"):
        config_from_mapping(
            base_config(splits={"probe_train": 0.5, "calibration": 0.3, "test": 0.3})
        )


def test_split_fractions_must_be_positive():
    with pytest.raises(ConfigError, match="positive"):
        config_from_mapping(
            base_config(splits={"probe_train": 1.0, "calibration": 0.0, "test": 0.0})
        )


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_alpha_must_be_a_probability(alpha):
    with pytest.raises(ConfigError, match="alpha"):
        config_from_mapping(base_config(conformal={"alpha_grid": [alpha], "delta": 0.1}))


def test_empty_alpha_grid_is_rejected():
    with pytest.raises(ConfigError, match="alpha_grid"):
        config_from_mapping(base_config(conformal={"alpha_grid": [], "delta": 0.1}))


def test_delta_must_be_a_probability():
    with pytest.raises(ConfigError, match="delta"):
        config_from_mapping(base_config(conformal={"alpha_grid": [0.05], "delta": 1.0}))


def test_missing_required_key_is_reported():
    with pytest.raises(ConfigError, match="model"):
        config_from_mapping(
            {"seed": 1, "splits": {"probe_train": 0.4, "calibration": 0.3, "test": 0.3},
             "conformal": {"alpha_grid": [0.05], "delta": 0.1}}
        )


def test_unknown_signal_field_is_rejected():
    with pytest.raises(ConfigError, match="unknown fields"):
        config_from_mapping(base_config(signals={"fields": ["total", "banana"]}))

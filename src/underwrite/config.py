"""Configuration loading and validation for Underwrite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from underwrite.extract.prompts import CANONICAL_FIELDS, FIELD_TYPES

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

_DEFAULT_LAYER_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)


class ConfigError(ValueError):
    """Raised when a configuration file is missing or contains invalid values."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str | None = None
    quantization: str = "nf4"
    compute_dtype: str = "bfloat16"
    max_new_tokens: int = 512
    do_sample: bool = False


@dataclass(frozen=True)
class SplitConfig:
    probe_train: float
    calibration: float
    test: float

    def __post_init__(self) -> None:
        total = self.probe_train + self.calibration + self.test
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"split fractions must sum to 1.0 (got {total:.6f})")
        if min(self.probe_train, self.calibration, self.test) <= 0.0:
            raise ConfigError("every split fraction must be positive")


@dataclass(frozen=True)
class ConformalConfig:
    alpha_grid: tuple[float, ...]
    delta: float
    threshold_grid_size: int = 501

    def __post_init__(self) -> None:
        if not self.alpha_grid:
            raise ConfigError("conformal.alpha_grid must not be empty")
        for alpha in self.alpha_grid:
            if not 0.0 < alpha < 1.0:
                raise ConfigError(f"alpha must lie strictly between 0 and 1 (got {alpha!r})")
        if not 0.0 < self.delta < 1.0:
            raise ConfigError(f"delta must lie strictly between 0 and 1 (got {self.delta!r})")
        if self.threshold_grid_size < 2:
            raise ConfigError("conformal.threshold_grid_size must be at least 2")


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    splits_file: str = "data/splits.json"


@dataclass(frozen=True)
class SignalsConfig:
    fields: tuple[str, ...] = CANONICAL_FIELDS
    hidden_layer_fractions: tuple[float, ...] = _DEFAULT_LAYER_FRACTIONS

    def __post_init__(self) -> None:
        if not self.fields:
            raise ConfigError("signals.fields must not be empty")
        unknown = set(self.fields) - set(FIELD_TYPES)
        if unknown:
            raise ConfigError(f"unknown fields in signals.fields: {sorted(unknown)}")
        for fraction in self.hidden_layer_fractions:
            if not 0.0 < fraction <= 1.0:
                raise ConfigError(f"hidden_layer_fractions must lie in (0, 1] (got {fraction!r})")


@dataclass(frozen=True)
class Config:
    seed: int
    model: ModelConfig
    splits: SplitConfig
    conformal: ConformalConfig
    data: DataConfig = field(default_factory=DataConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in {context}")
    return mapping[key]


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping")
    return value


def config_from_mapping(raw: Mapping[str, Any]) -> Config:
    """Build a :class:`Config` from an already-parsed mapping, applying defaults."""
    model_raw = _section(raw, "model")
    splits_raw = _section(raw, "splits")
    conformal_raw = _section(raw, "conformal")
    data_raw = _section(raw, "data")
    signals_raw = _section(raw, "signals")

    model = ModelConfig(
        id=_require(model_raw, "id", "model"),
        revision=model_raw.get("revision"),
        quantization=model_raw.get("quantization", "nf4"),
        compute_dtype=model_raw.get("compute_dtype", "bfloat16"),
        max_new_tokens=int(model_raw.get("max_new_tokens", 512)),
        do_sample=bool(model_raw.get("do_sample", False)),
    )
    splits = SplitConfig(
        probe_train=float(_require(splits_raw, "probe_train", "splits")),
        calibration=float(_require(splits_raw, "calibration", "splits")),
        test=float(_require(splits_raw, "test", "splits")),
    )
    conformal = ConformalConfig(
        alpha_grid=tuple(float(a) for a in _require(conformal_raw, "alpha_grid", "conformal")),
        delta=float(_require(conformal_raw, "delta", "conformal")),
        threshold_grid_size=int(conformal_raw.get("threshold_grid_size", 501)),
    )
    data = DataConfig(
        raw_dir=data_raw.get("raw_dir", "data/raw"),
        processed_dir=data_raw.get("processed_dir", "data/processed"),
        cache_dir=data_raw.get("cache_dir", "data/cache"),
        splits_file=data_raw.get("splits_file", "data/splits.json"),
    )
    signals = SignalsConfig(
        fields=tuple(signals_raw.get("fields", CANONICAL_FIELDS)),
        hidden_layer_fractions=tuple(
            float(f) for f in signals_raw.get("hidden_layer_fractions", _DEFAULT_LAYER_FRACTIONS)
        ),
    )

    return Config(
        seed=int(_require(raw, "seed", "config")),
        model=model,
        splits=splits,
        conformal=conformal,
        data=data,
        signals=signals,
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML configuration file."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, Mapping):
        raise ConfigError(f"config root must be a mapping: {config_path}")

    return config_from_mapping(raw)

"""Probe and fusion baselines.

Each method turns a document's signals into one confidence score per field, so that
a selective predictor can rank fields and auto-accept the top ones.

Fitting rules, from the statistical design in the plan:

* every model is fitted on the **probe_train** split only;
* the probe's layer, PCA width and regularisation are chosen by document-level
  cross-validation *inside* probe_train;
* the fusion model that consumes the probe score is trained on **out-of-fold**
  probe scores, so the probe score it sees during fitting is not one it produced
  for its own training rows;
* the test split is scored once, at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from underwrite.eval.risk_coverage import summarise
from underwrite.probe.features import load_features
from underwrite.signals.validators import CHECK_NAMES

TRAIN_SPLIT = "probe_train"
TEST_SPLIT = "test"

# Token-level features; entropy is negated so that "higher is more correct" holds.
TOKEN_COLUMNS = ("mean_logprob", "min_logprob", "first_token_entropy", "first_token_margin", "n_tokens")
BLACK_BOX_COLUMNS = TOKEN_COLUMNS + tuple(CHECK_NAMES) + ("grounding", "verbalized")

PCA_COMPONENTS = (32, 64)
REGULARISATION = (0.01, 0.1, 1.0)
CV_FOLDS = 5


@dataclass(frozen=True)
class ProbeChoice:
    """The hyper-parameters cross-validation selected for the hidden-state probe."""

    layer: int
    n_components: int
    C: float
    cv_auroc: float


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame[list(columns)].to_numpy(dtype=float).copy()
    if "first_token_entropy" in columns:
        index = list(columns).index("first_token_entropy")
        values[:, index] = -values[:, index]
    return values


def _logistic_pipeline(**overrides: Any) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if "n_components" in overrides:
        steps.append(("scale_pca", StandardScaler()))
        steps.append(("pca", PCA(n_components=overrides.pop("n_components"), random_state=0)))
    else:
        steps.append(("scale", StandardScaler()))
    steps.append(
        ("model", LogisticRegression(max_iter=1000, C=overrides.pop("C", 1.0), random_state=0))
    )
    return Pipeline(steps)


def _probabilities(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.predict_proba(X)[:, 1]


def _hidden_matrix(
    features_dir: str | Path,
    frame: pd.DataFrame,
    field: str,
    layer: int,
    hidden_size: int | None = None,
) -> np.ndarray:
    """Stack the last-token hidden vector for each row, zero-filling any gaps."""
    root = Path(features_dir)
    rows: list[np.ndarray] = []
    for doc_id in frame["doc_id"]:
        path = root / f"{doc_id}.npz"
        vector: np.ndarray | None = None
        if path.is_file():
            arrays, _ = load_features(path)
            key = f"hidden__{field}__L{layer}"
            if key in arrays:
                vector = arrays[key][0].astype(np.float32)
        if vector is None:
            width = hidden_size or 1536
            vector = np.zeros(width, dtype=np.float32)
        rows.append(vector)
    return np.vstack(rows)


def select_probe(
    features_dir: str | Path,
    frame: pd.DataFrame,
    field: str,
    labels: np.ndarray,
    layers: Sequence[int],
    seed: int = 0,
) -> ProbeChoice:
    """Choose layer, PCA width and regularisation by cross-validation on train."""
    splits = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    best: ProbeChoice | None = None

    for layer in layers:
        X = _hidden_matrix(features_dir, frame, field, layer)
        for n_components in PCA_COMPONENTS:
            for C in REGULARISATION:
                pipeline = _logistic_pipeline(n_components=n_components, C=C)
                try:
                    probabilities = cross_val_predict(
                        pipeline, X, labels, cv=splits, method="predict_proba"
                    )[:, 1]
                except ValueError:
                    continue
                score = float(roc_auc_score(labels, probabilities))
                if best is None or score > best.cv_auroc:
                    best = ProbeChoice(layer=layer, n_components=n_components, C=C, cv_auroc=score)

    if best is None:
        raise RuntimeError(f"no usable probe configuration for field {field!r}")
    return best


def run_field(
    frame: pd.DataFrame,
    field: str,
    features_dir: str | Path,
    layers: Sequence[int],
    seed: int = 0,
) -> pd.DataFrame:
    """Score every method for one field and return its metrics table."""
    subset = frame[frame["field"] == field].reset_index(drop=True)
    train = subset[subset["split"] == TRAIN_SPLIT].reset_index(drop=True)
    test = subset[subset["split"] == TEST_SPLIT].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"field {field!r} has no train or test rows")

    y_train = train["correct"].to_numpy(dtype=int)
    y_test = test["correct"].to_numpy(dtype=int)
    errors = 1 - y_test

    results: dict[str, np.ndarray] = {}

    # S0: mean token log-probability.
    results["S0 logprob"] = test["mean_logprob"].fillna(
        train["mean_logprob"].median()
    ).to_numpy(dtype=float)

    # S1: verbalized self-rating.
    results["S1 verbalized"] = test["verbalized"].fillna(
        train["verbalized"].median()
    ).to_numpy(dtype=float)

    # S2: validators only.
    X_val_train = _matrix(train, CHECK_NAMES)
    X_val_test = _matrix(test, CHECK_NAMES)
    validator_model = _logistic_pipeline().fit(X_val_train, y_train)
    results["S2 validators"] = _probabilities(validator_model, X_val_test)

    # S2g: source grounding alone.
    results["S2g grounding"] = test["grounding"].fillna(0.0).to_numpy(dtype=float)

    # S3: the strongest black-box fusion (no hidden states).
    X_bb_train = _matrix(train, BLACK_BOX_COLUMNS)
    X_bb_test = _matrix(test, BLACK_BOX_COLUMNS)
    black_box_model = _logistic_pipeline().fit(X_bb_train, y_train)
    results["S3 black-box fusion"] = _probabilities(black_box_model, X_bb_test)

    # S4: hidden-state probe only.
    choice = select_probe(features_dir, train, field, y_train, layers, seed=seed)
    probe_train = _hidden_matrix(features_dir, train, field, choice.layer)
    probe_test = _hidden_matrix(features_dir, test, field, choice.layer)
    probe_pipeline = _logistic_pipeline(n_components=choice.n_components, C=choice.C)
    probe_pipeline.fit(probe_train, y_train)
    results["S4 probe"] = _probabilities(probe_pipeline, probe_test)

    # S5: fusion of the black-box features with an out-of-fold probe score.
    splits = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    oof_probe = cross_val_predict(
        _logistic_pipeline(n_components=choice.n_components, C=choice.C),
        probe_train,
        y_train,
        cv=splits,
        method="predict_proba",
    )[:, 1]
    fusion_train = np.column_stack([X_bb_train, oof_probe])
    fusion_test = np.column_stack([X_bb_test, results["S4 probe"]])
    fusion_model = _logistic_pipeline().fit(fusion_train, y_train)
    results["S5 probe + fusion"] = _probabilities(fusion_model, fusion_test)

    rows = []
    for method, scores in results.items():
        metrics = summarise(scores, errors)
        rows.append(
            {
                "field": field,
                "method": method,
                "n_test": len(test),
                "base_error": float(np.mean(errors)),
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["probe_choice"] = choice
    return table

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

Scoring is split into :func:`fit_scorers` and :func:`apply_scorers` so that the same
fitted objects can score a *different* corpus, which is what the shift experiment
needs: fit on CORD, deploy on synthetic invoices.
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

TOKEN_COLUMNS = (
    "mean_logprob",
    "min_logprob",
    "first_token_entropy",
    "first_token_margin",
    "n_tokens",
)
BLACK_BOX_COLUMNS = TOKEN_COLUMNS + tuple(CHECK_NAMES) + ("grounding", "verbalized")

METHOD_COLUMNS = (
    "S0 logprob",
    "S1 verbalized",
    "S2 validators",
    "S2g grounding",
    "S3 black-box fusion",
    "S4 probe",
    "S5 probe + fusion",
)

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


@dataclass
class Scorers:
    """All fitted objects, reusable on an unseen corpus."""

    field: str
    choice: ProbeChoice
    logprob_fill: float
    verbalized_fill: float
    validators: Any
    black_box: Any
    probe: Any
    fusion: Any


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame[list(columns)].to_numpy(dtype=float).copy()
    if "first_token_entropy" in columns:
        # Entropy points the other way: more entropy means less confidence.
        values[:, list(columns).index("first_token_entropy")] *= -1.0
    # A corpus captured without one of the signals (verbalized, say) leaves a column
    # entirely missing. Collapse it to a constant so the imputer and the model see
    # finite numbers instead of NaN.
    medians = np.nanmedian(values, axis=0) if values.size else np.array([])
    for index in range(values.shape[1]):
        if np.isnan(medians[index]):
            values[:, index] = 0.0
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
    hidden_size: int = 1536,
) -> np.ndarray:
    """Stack the last-token hidden vector for each row, zero-filling any gaps."""
    root = Path(features_dir)
    rows: list[np.ndarray] = []
    for doc_id in frame["doc_id"]:
        vector: np.ndarray | None = None
        path = root / f"{doc_id}.npz"
        if path.is_file():
            arrays, _ = load_features(path)
            key = f"hidden__{field}__L{layer}"
            if key in arrays:
                vector = arrays[key][0].astype(np.float32)
        rows.append(vector if vector is not None else np.zeros(hidden_size, dtype=np.float32))
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
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    best: ProbeChoice | None = None

    for layer in layers:
        X = _hidden_matrix(features_dir, frame, field, layer)
        for n_components in PCA_COMPONENTS:
            for C in REGULARISATION:
                pipeline = _logistic_pipeline(n_components=n_components, C=C)
                try:
                    probabilities = cross_val_predict(
                        pipeline, X, labels, cv=folds, method="predict_proba"
                    )[:, 1]
                except ValueError:
                    continue
                score = float(roc_auc_score(labels, probabilities))
                if best is None or score > best.cv_auroc:
                    best = ProbeChoice(layer, n_components, C, score)

    if best is None:
        raise RuntimeError(f"no usable probe configuration for field {field!r}")
    return best


def fit_scorers(
    train: pd.DataFrame,
    field: str,
    features_dir: str | Path,
    layers: Sequence[int],
    seed: int = 0,
) -> Scorers:
    """Fit every method on the given training rows."""
    labels = train["correct"].to_numpy(dtype=int)
    choice = select_probe(features_dir, train, field, labels, layers, seed=seed)

    probe_train = _hidden_matrix(features_dir, train, field, choice.layer)
    probe_pipeline = _logistic_pipeline(n_components=choice.n_components, C=choice.C)
    probe_pipeline.fit(probe_train, labels)

    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    out_of_fold = cross_val_predict(
        _logistic_pipeline(n_components=choice.n_components, C=choice.C),
        probe_train,
        labels,
        cv=folds,
        method="predict_proba",
    )[:, 1]

    black_box = _logistic_pipeline().fit(_matrix(train, BLACK_BOX_COLUMNS), labels)
    fusion = _logistic_pipeline().fit(
        np.column_stack([_matrix(train, BLACK_BOX_COLUMNS), out_of_fold]), labels
    )
    validators = _logistic_pipeline().fit(_matrix(train, CHECK_NAMES), labels)

    # A corpus captured without the self-rating call has no verbalized scores at all.
    # Fall back to a flat 0.5 so S1 degrades to an uninformative baseline rather than
    # poisoning every downstream metric with NaN.
    logprob_fill = train["mean_logprob"].median()
    verbalized_fill = train["verbalized"].median()

    return Scorers(
        field=field,
        choice=choice,
        logprob_fill=float(logprob_fill) if pd.notna(logprob_fill) else 0.0,
        verbalized_fill=float(verbalized_fill) if pd.notna(verbalized_fill) else 0.5,
        validators=validators,
        black_box=black_box,
        probe=probe_pipeline,
        fusion=fusion,
    )


def apply_scorers(
    scorers: Scorers, frame: pd.DataFrame, features_dir: str | Path
) -> pd.DataFrame:
    """Score an arbitrary frame with already-fitted scorers."""
    scores: dict[str, np.ndarray] = {}
    scores["S0 logprob"] = frame["mean_logprob"].fillna(scorers.logprob_fill).to_numpy(float)
    scores["S1 verbalized"] = frame["verbalized"].fillna(scorers.verbalized_fill).to_numpy(float)
    scores["S2 validators"] = _probabilities(scorers.validators, _matrix(frame, CHECK_NAMES))
    scores["S2g grounding"] = frame["grounding"].fillna(0.0).to_numpy(float)
    scores["S3 black-box fusion"] = _probabilities(
        scorers.black_box, _matrix(frame, BLACK_BOX_COLUMNS)
    )

    probe = _hidden_matrix(features_dir, frame, scorers.field, scorers.choice.layer)
    scores["S4 probe"] = _probabilities(scorers.probe, probe)
    scores["S5 probe + fusion"] = _probabilities(
        scorers.fusion,
        np.column_stack([_matrix(frame, BLACK_BOX_COLUMNS), scores["S4 probe"]]),
    )

    scored = pd.DataFrame(
        {"doc_id": frame["doc_id"], "split": frame["split"], "correct": frame["correct"]}
    )
    for method in METHOD_COLUMNS:
        scored[method] = scores[method]
    return scored


def score_all(
    frame: pd.DataFrame,
    field: str,
    features_dir: str | Path,
    layers: Sequence[int],
    seed: int = 0,
) -> tuple[pd.DataFrame, ProbeChoice]:
    """Fit on probe_train and score every row of ``field`` in the same corpus."""
    subset = frame[frame["field"] == field].reset_index(drop=True)
    train = subset[subset["split"] == TRAIN_SPLIT].reset_index(drop=True)
    if train.empty:
        raise ValueError(f"field {field!r} has no probe_train rows")
    scorers = fit_scorers(train, field, features_dir, layers, seed=seed)
    return apply_scorers(scorers, subset, features_dir), scorers.choice


def run_field(
    frame: pd.DataFrame,
    field: str,
    features_dir: str | Path,
    layers: Sequence[int],
    seed: int = 0,
) -> pd.DataFrame:
    """Score every method for one field and return its metrics on the test split."""
    scored, choice = score_all(frame, field, features_dir, layers, seed=seed)
    test = scored[scored["split"] == TEST_SPLIT]
    if test.empty:
        raise ValueError(f"field {field!r} has no test rows")

    errors = 1 - test["correct"].to_numpy(dtype=int)
    table = pd.DataFrame(
        [
            {
                "field": field,
                "method": method,
                "n_test": len(test),
                "base_error": float(errors.mean()),
                **summarise(test[method].to_numpy(dtype=float), errors),
            }
            for method in METHOD_COLUMNS
        ]
    )
    table.attrs["probe_choice"] = choice
    return table

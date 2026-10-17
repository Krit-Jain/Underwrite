import math

import numpy as np
import pytest

from underwrite.extract.vlm import TeacherForcedOutput
from underwrite.probe.features import (
    build_document_features,
    load_features,
    save_features,
)
from underwrite.signals.teacher_forced import FEATURE_NAMES
from underwrite.signals.validators import CHECK_NAMES

FIELDS = ("vendor", "date", "subtotal", "tax", "total")
LAYERS = (9, 18)

TEXT = '{"vendor": "ACME STORES", "date": null, "subtotal": "100.00", "tax": "18.00", "total": "118.00"}'


def one_token_per_character():
    """Character-level offsets, so token spans are exact and deterministic."""
    return tuple((index, index + 1) for index in range(len(TEXT)))


def make_signals(hidden_size: int = 8) -> TeacherForcedOutput:
    offsets = one_token_per_character()
    width = len(offsets)
    hidden = {
        layer: np.full((width, hidden_size), fill_value=layer, dtype=np.float16)
        for layer in LAYERS
    }
    return TeacherForcedOutput(
        text=TEXT,
        n_prompt_tokens=10,
        token_ids=tuple(range(width)),
        offsets=offsets,
        logprobs=(-0.2,) * width,
        entropies=(0.5,) * width,
        margins=(1.5,) * width,
        hidden_states=hidden,
        offsets_aligned=True,
    )


def test_captures_an_entry_for_every_field_present_in_the_text():
    features = build_document_features("doc_0", {"total": "118.00"}, {}, make_signals(), FIELDS)
    assert set(features.fields) == {"vendor", "date", "subtotal", "tax", "total"}


def test_a_null_value_still_produces_signals():
    # How confidently the model said "no value" is itself a useful signal.
    features = build_document_features("doc_0", {}, {}, make_signals(), FIELDS)
    assert features.fields["date"].token["n_tokens"] == len("null")


def test_token_statistics_cover_the_whole_value():
    features = build_document_features("doc_0", {}, {}, make_signals(), FIELDS)
    assert features.fields["subtotal"].token["n_tokens"] == len("100.00")
    assert features.fields["subtotal"].token["mean_logprob"] == pytest.approx(-0.2)


def test_hidden_states_keep_last_and_mean_for_each_layer():
    features = build_document_features("doc_0", {}, {}, make_signals(), FIELDS)
    hidden = features.fields["total"].hidden
    assert set(hidden) == set(LAYERS)
    assert hidden[9].shape == (2, 8)
    assert hidden[9].dtype == np.float16
    assert np.allclose(hidden[18].astype(float), 18.0)


def test_missing_fields_produce_nan_rows():
    features = build_document_features("doc_0", {}, {}, make_signals(), ("total", "vendor"))
    features.fields.pop("vendor")
    row = features.token_row("vendor")
    assert len(row) == len(FEATURE_NAMES)
    assert all(math.isnan(value) for value in row)


def test_save_and_load_round_trip(tmp_path):
    signals = make_signals()
    features = build_document_features(
        "doc_7", {"total": "118.00"}, {"total": "118.00"}, signals, FIELDS, {"sum_matches": 1.0}
    )
    path = save_features(tmp_path / "nested" / "doc_7.npz", features, FIELDS, LAYERS)

    arrays, meta = load_features(path)
    assert meta["doc_id"] == "doc_7"
    assert meta["extracted"] == {"total": "118.00"}
    assert arrays["token__total"].shape == (len(FEATURE_NAMES),)
    assert arrays["hidden__total__L9"].shape == (2, 8)
    assert arrays["validators"].shape == (len(CHECK_NAMES),)
    assert meta["gold"] == {"total": "118.00"}


def test_saved_validators_keep_the_canonical_order(tmp_path):
    features = build_document_features("doc_1", {}, {}, make_signals(), FIELDS, {"sum_matches": 1.0})
    path = save_features(tmp_path / "doc_1.npz", features, FIELDS, LAYERS)
    arrays, meta = load_features(path)
    index = meta["check_names"].index("sum_matches")
    assert arrays["validators"][index] == 1.0

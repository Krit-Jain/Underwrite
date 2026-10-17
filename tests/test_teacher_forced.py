import math
from functools import lru_cache

import pytest

from underwrite.signals.teacher_forced import (
    FEATURE_NAMES,
    as_array,
    entropy,
    field_statistics,
    flatten,
    locate_field_values,
    select_tokens,
    summarise_fields,
    top_margin,
)

SAMPLE = (
    "{\n"
    '  "vendor": "KRISHNA HARDWARE MART",\n'
    '  "date": "2025-12-21",\n'
    '  "subtotal": "8301.50",\n'
    '  "tax": "498.09",\n'
    '  "total": "9297.68"\n'
    "}"
)

FIELDS = ("vendor", "date", "subtotal", "tax", "total")


def value_of(text, spans, field):
    low, high = spans[field]
    return text[low:high]


def test_string_values_exclude_their_quotes():
    spans = locate_field_values(SAMPLE, FIELDS)
    assert value_of(SAMPLE, spans, "vendor") == "KRISHNA HARDWARE MART"
    assert value_of(SAMPLE, spans, "total") == "9297.68"
    assert '"' not in value_of(SAMPLE, spans, "vendor")


def test_bare_numeric_values_are_located():
    text = '{"total": 354.00, "tax": null}'
    spans = locate_field_values(text, ("total", "tax"))
    assert text[slice(*spans["total"])] == "354.00"
    assert text[slice(*spans["tax"])] == "null"


def test_trailing_whitespace_is_trimmed_from_bare_values():
    text = '{"total": 354.00   }'
    spans = locate_field_values(text, ("total",))
    assert text[slice(*spans["total"])] == "354.00"


def test_escaped_quotes_do_not_end_the_string():
    text = r'{"vendor": "Bob\"s Cafe", "total": 5.0}'
    spans = locate_field_values(text, ("vendor", "total"))
    assert text[slice(*spans["vendor"])] == r'Bob\"s Cafe'
    assert text[slice(*spans["total"])] == "5.0"


def test_absent_fields_are_omitted():
    spans = locate_field_values('{"total": 1.0}', FIELDS)
    assert set(spans) == {"total"}


def test_empty_string_values_are_omitted():
    spans = locate_field_values('{"vendor": "", "total": 1.0}', FIELDS)
    assert "vendor" not in spans


def test_unclosed_string_does_not_crash():
    spans = locate_field_values('{"vendor": "unterminated', ("vendor",))
    assert spans["vendor"] == (12, 24)


def test_select_tokens_uses_overlap():
    offsets = [(0, 3), (3, 6), (6, 9)]
    assert select_tokens(offsets, (4, 7)) == [1, 2]
    assert select_tokens(offsets, (0, 1)) == [0]


def test_select_tokens_never_picks_special_tokens():
    offsets = [(0, 0), (0, 5), (5, 9)]
    assert select_tokens(offsets, (0, 5)) == [1]


@pytest.mark.parametrize(
    "log_prob,expected",
    [(math.log(1.0), 0.0), (math.log(0.5), -0.5 * math.log(0.5))],
)
def test_entropy_of_a_single_outcome(log_prob, expected):
    # A one-entry vector is a distribution over one outcome, so H = -p ln p.
    assert entropy([log_prob]) == pytest.approx(expected)


def test_entropy_of_a_uniform_pair_is_ln_two():
    assert entropy([math.log(0.5), math.log(0.5)]) == pytest.approx(math.log(2))


def test_entropy_ignores_impossible_outcomes():
    assert entropy([math.log(1.0), -math.inf]) == pytest.approx(0.0)


def test_top_margin_is_the_gap_between_the_two_best():
    assert top_margin([math.log(0.7), math.log(0.2), math.log(0.1)]) == pytest.approx(math.log(3.5))


def test_top_margin_needs_two_outcomes():
    assert top_margin([0.0]) == 0.0


def test_field_statistics_aggregates_value_tokens():
    summary = field_statistics([-0.1, -0.5, -0.2], [0.9, 1.1, 1.0], [2.0, 3.0, 4.0])
    assert summary.n_tokens == 3
    assert summary.mean_logprob == pytest.approx(-0.26666666)
    assert summary.min_logprob == -0.5
    assert summary.first_token_entropy == 0.9
    assert summary.first_token_margin == 2.0


def test_field_statistics_is_none_without_tokens():
    assert field_statistics([], [], []) is None


def test_flatten_prefixes_each_feature_with_its_field():
    summary = field_statistics([-1.0], [0.5], [1.5])
    flat = flatten({"total": summary})
    assert flat["total.mean_logprob"] == -1.0
    assert flat["total.n_tokens"] == 1.0


def test_as_array_keeps_field_order_and_marks_missing_fields():
    summary = field_statistics([-1.0], [0.5], [1.5])
    rows = as_array({"total": summary}, ("vendor", "total"))
    assert list(rows) == ["vendor", "total"]
    assert len(rows["total"]) == len(FEATURE_NAMES)
    assert all(math.isnan(value) for value in rows["vendor"])


@lru_cache(maxsize=1)
def _gemma_tokenizer():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained("google/gemma-4-E2B-it").tokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_token_selection_reconstructs_every_value_exactly():
    """The real tokenizer must map each value span back to exactly that value."""
    tokenizer = _gemma_tokenizer()
    encoding = tokenizer(SAMPLE, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    spans = locate_field_values(SAMPLE, FIELDS)

    for field, span in spans.items():
        indices = select_tokens(offsets, span)
        assert indices, f"no tokens selected for {field}"
        reconstructed = "".join(
            SAMPLE[max(start, span[0]) : min(end, span[1])]
            for start, end in (offsets[index] for index in indices)
        )
        assert reconstructed == value_of(SAMPLE, spans, field)


def test_summarise_fields_runs_over_a_real_tokenizer():
    tokenizer = _gemma_tokenizer()
    encoding = tokenizer(SAMPLE, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    width = len(offsets)

    statistics = summarise_fields(
        SAMPLE,
        offsets,
        [-0.2] * width,
        [0.8] * width,
        [1.5] * width,
        FIELDS,
    )
    assert set(statistics) == set(FIELDS)
    for summary in statistics.values():
        assert summary.n_tokens >= 1
        assert summary.mean_logprob == pytest.approx(-0.2)

import pytest

from underwrite.data.splits import (
    SplitError,
    assign_splits,
    load_splits,
    summarize,
    write_splits,
)


def make_ids(count: int) -> list[str]:
    return [f"doc_{index:04d}" for index in range(count)]


def test_every_document_is_assigned_exactly_once():
    ids = make_ids(100)
    assignment = assign_splits(ids, seed=1)
    assert set(assignment) == set(ids)
    assert set(assignment.values()) <= {"probe_train", "calibration", "test"}


def test_assignment_is_deterministic_for_a_seed():
    ids = make_ids(50)
    assert assign_splits(ids, seed=7) == assign_splits(ids, seed=7)


def test_different_seeds_shuffle_differently():
    ids = make_ids(50)
    assert assign_splits(ids, seed=1) != assign_splits(ids, seed=2)


def test_input_order_does_not_matter():
    ids = make_ids(30)
    assert assign_splits(ids, seed=3) == assign_splits(list(reversed(ids)), seed=3)


def test_split_sizes_follow_the_fractions():
    counts = summarize(assign_splits(make_ids(100), seed=5))
    assert counts == {"probe_train": 40, "calibration": 30, "test": 30}


def test_counts_always_sum_to_the_corpus_size():
    for size in (1, 2, 3, 7, 99, 1000):
        counts = summarize(assign_splits(make_ids(size), seed=11))
        assert sum(counts.values()) == size


def test_fractions_must_sum_to_one():
    with pytest.raises(SplitError, match="sum to 1.0"):
        assign_splits(make_ids(10), probe_train=0.5, calibration=0.3, test=0.3)


def test_negative_fractions_are_rejected():
    with pytest.raises(SplitError, match="non-negative"):
        assign_splits(make_ids(10), probe_train=-0.1, calibration=0.6, test=0.5)


def test_write_then_load_round_trips(tmp_path):
    assignment = assign_splits(make_ids(20), seed=2)
    path = write_splits(tmp_path / "nested" / "splits.json", assignment)
    assert path.is_file()
    assert load_splits(path) == assignment

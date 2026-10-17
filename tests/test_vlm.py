import pytest

from underwrite.extract.vlm import _find_subsequence, pick_layers


def test_finds_a_subsequence_in_the_middle():
    assert _find_subsequence([1, 2, 3, 4, 5], [3, 4]) == 2


def test_finds_a_subsequence_at_the_start():
    assert _find_subsequence([1, 2, 3], [1, 2]) == 0


def test_returns_none_when_absent():
    assert _find_subsequence([1, 2, 3], [9, 9]) is None


def test_empty_needle_is_not_a_match():
    assert _find_subsequence([1, 2, 3], []) is None


def test_needle_longer_than_haystack_is_not_a_match():
    assert _find_subsequence([1], [1, 2]) is None


def test_returns_the_first_occurrence():
    assert _find_subsequence([7, 7, 7], [7, 7]) == 0


@pytest.mark.parametrize("fraction,count,expected", [(0.25, 36, 9), (0.5, 36, 18), (1.0, 36, 35)])
def test_depth_fractions_map_onto_hidden_state_indices(fraction, count, expected):
    assert pick_layers(count, (fraction,)) == [expected]


def test_the_configured_defaults_pick_four_layers():
    assert pick_layers(36, (0.25, 0.5, 0.75, 0.9)) == [9, 18, 26, 32]


def test_embedding_output_is_never_selected():
    # hidden_states[0] is the embedding output, which is not a transformer layer.
    assert 0 not in pick_layers(36, (0.0, 0.01))


def test_layers_are_sorted_and_deduplicated():
    layers = pick_layers(36, (0.9, 0.25, 0.9))
    assert layers == sorted(set(layers))

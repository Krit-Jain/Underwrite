import pytest

from underwrite.signals.gstin import (
    build_gstin,
    check_digit,
    is_valid_gstin,
    normalize_gstin,
)

# Cases published alongside the algorithm, chosen to break naive implementations.
VALID_GSTINS = [
    ("27AAACR5055K1Z7", "real registration"),
    ("29AAACI1195H1ZI", "check digit is a letter"),
    ("07AABCU9603R1ZP", "another letter check digit"),
    ("99AAACR5055KAZN", "entity code A, state code 99"),
    ("00AAACR5055K1ZN", "correct checksum, state code never issued"),
    ("27aaacr5055k1z7", "lowercase must normalize"),
    ("27-AAACR-5055K1Z7", "pasted punctuation must normalize"),
]

INVALID_GSTINS = [
    ("27AAACR5055K1Z0", "check digit off by one"),
    ("27AAACR5055K1Y7", "position 14 is not Z"),
    ("27AAACR5055K0Z7", "entity code 0 is never issued"),
    ("27AAACR50X5K1Z7", "letter inside the PAN digit block"),
    ("271234R5055K1Z7", "digits inside the PAN letter block"),
    ("27AAACR5055K1Z", "too short"),
    ("", "empty"),
]


@pytest.mark.parametrize("value,reason", VALID_GSTINS)
def test_valid_gstins(value, reason):
    assert is_valid_gstin(value), reason


@pytest.mark.parametrize("value,reason", INVALID_GSTINS)
def test_invalid_gstins(value, reason):
    assert not is_valid_gstin(value), reason


def test_worked_example_from_the_specification():
    assert check_digit("27AAACR5055K1Z") == "7"


@pytest.mark.parametrize(
    "body,expected",
    [("29AAACI1195H1Z", "I"), ("07AABCU9603R1Z", "P")],
)
def test_letter_check_digits(body, expected):
    assert check_digit(body) == expected


def test_normalize_gstin_strips_noise():
    assert normalize_gstin(" 27-aaacr 5055k1z7 ") == "27AAACR5055K1Z7"
    assert normalize_gstin(None) == ""


def test_never_issued_state_code_is_a_separate_fact():
    assert is_valid_gstin("00AAACR5055K1ZN")
    assert not is_valid_gstin("00AAACR5055K1ZN", require_known_state=True)


def test_build_gstin_matches_the_published_example():
    assert build_gstin("AAACR5055K", "27") == "27AAACR5055K1Z7"


def test_build_gstin_output_is_always_valid():
    for state in ("07", "27", "29", "33"):
        assert is_valid_gstin(build_gstin("ABCUE9603R", state), require_known_state=True)

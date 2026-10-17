"""GSTIN structure and check-digit validation for Indian tax invoices.

The check digit is a modulus-36 variant of the Luhn algorithm over the first
fourteen characters. It catches every single-character typo, but it says nothing
about whether a registration exists or is still active.
"""

from __future__ import annotations

import re

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# state code (2 digits) | PAN (5 letters, 4 digits, 1 letter) | entity code (never 0)
# | reserved 'Z' | check digit
GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")

# State and union-territory codes issued under the current scheme.
KNOWN_STATE_CODES = frozenset([f"{code:02d}" for code in range(1, 39)] + ["96", "97", "99"])

_NON_ALNUM = re.compile(r"[^0-9A-Za-z]")


def normalize_gstin(value: str | None) -> str:
    """Uppercase a GSTIN and strip separators people paste in (spaces, dashes)."""
    if value is None:
        return ""
    return _NON_ALNUM.sub("", value).upper()


def check_digit(first_fourteen: str) -> str:
    """Modulus-36 Luhn check digit for the first fourteen characters."""
    total = 0
    for index, character in enumerate(first_fourteen[:14]):
        product = CHARSET.index(character) * (1 if index % 2 == 0 else 2)
        total += product // 36 + product % 36
    return CHARSET[(36 - total % 36) % 36]


def is_valid_gstin(value: str | None, *, require_known_state: bool = False) -> bool:
    """Return whether ``value`` is a structurally valid, checksum-correct GSTIN.

    A correct checksum does not prove a registration exists. Set
    ``require_known_state`` to additionally reject state codes that are never
    issued (for example ``00``), which the checksum happily accepts.
    """
    gstin = normalize_gstin(value)
    if not GSTIN_PATTERN.match(gstin):
        return False
    if check_digit(gstin[:14]) != gstin[14]:
        return False
    if require_known_state and gstin[:2] not in KNOWN_STATE_CODES:
        return False
    return True


def build_gstin(pan: str, state_code: str, entity_code: str = "1") -> str:
    """Assemble a checksum-correct GSTIN from a PAN and a state code."""
    body = f"{state_code}{pan}{entity_code}Z"
    return body + check_digit(body)

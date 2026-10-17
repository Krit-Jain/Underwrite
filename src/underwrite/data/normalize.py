"""Normalization and correctness checks for extracted fields.

Money values arrive in mixed formats. Indonesian rupiah uses both ``.`` and ``,``
as thousands separators and has no minor unit, while Indian invoices use ``.`` for
paise. ``parse_money`` infers the separator by default and accepts an explicit
``decimal_sep`` to pin a single convention for a whole corpus.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from rapidfuzz import fuzz

MONEY_FIELDS = ("subtotal", "tax", "total")
DATE_FIELDS = ("date",)
VENDOR_FIELDS = ("vendor",)

_CURRENCY_WORDS = re.compile(r"\b(?:rp|idr|inr|usd|eur|gbp|rs)\b", re.IGNORECASE)
_NON_NUMERIC = re.compile(r"[^0-9.,-]")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


def _infer_decimal_sep(cleaned: str) -> str:
    """Infer the decimal separator, treating a 3-digit tail as a thousands group."""
    dot = cleaned.rfind(".")
    comma = cleaned.rfind(",")
    if dot != -1 and comma != -1:
        return "." if dot > comma else ","
    if dot == -1 and comma == -1:
        return ""
    sep = "." if dot != -1 else ","
    head, tail = cleaned.rsplit(sep, 1)
    if len(tail) == 3 and (cleaned.count(sep) > 1 or len(head) <= 3):
        return ""
    return sep


def parse_money(value: Any, decimal_sep: str | None = None) -> Decimal | None:
    """Parse a money value to :class:`~decimal.Decimal`, or ``None`` if unreadable.

    ``decimal_sep`` may be ``"."`` or ``","`` to pin the decimal separator, or the
    empty string to treat every separator as a thousands group (rupiah style).
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))

    cleaned = _CURRENCY_WORDS.sub("", str(value)).strip()
    cleaned = _NON_NUMERIC.sub("", cleaned)
    if not cleaned or cleaned.strip("-.,") == "":
        return None

    negative = cleaned.startswith("-")
    cleaned = cleaned.lstrip("-")
    cleaned = cleaned.strip(".,")
    if not cleaned:
        return None

    if decimal_sep is None:
        decimal_sep = _infer_decimal_sep(cleaned)
    if decimal_sep == "":
        digits = cleaned.replace(".", "").replace(",", "")
    elif decimal_sep == ".":
        digits = cleaned.replace(",", "")
    elif decimal_sep == ",":
        digits = cleaned.replace(".", "").replace(",", ".")
    else:
        raise ValueError(f"decimal_sep must be '.', ',' or '', got {decimal_sep!r}")

    try:
        parsed = Decimal(digits)
    except InvalidOperation:
        return None
    return -parsed if negative else parsed


def normalize_date(value: Any) -> date | None:
    """Parse a date to :class:`datetime.date`, or ``None`` if unreadable.

    Ambiguous ``DD/MM/YYYY`` is read day-first, which matches the corpora used here.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_vendor(value: Any) -> str:
    """Case-fold a vendor name and reduce it to space-separated alphanumerics."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return _NON_ALNUM.sub(" ", text).strip()


def vendor_matches(left: Any, right: Any, threshold: float = 90.0) -> bool:
    """Whether two vendor names agree after normalization."""
    a, b = normalize_vendor(left), normalize_vendor(right)
    if not a or not b:
        return False
    return fuzz.ratio(a, b) >= threshold


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def is_correct(
    field: str,
    predicted: Any,
    gold: Any,
    *,
    decimal_sep: str | None = None,
    vendor_threshold: float = 90.0,
) -> bool:
    """Whether a predicted field value matches the gold value.

    A predicted ``None`` against a present gold value is an error, so a parse
    failure is never silently scored as correct.
    """
    if _is_blank(gold):
        return _is_blank(predicted)

    if field in MONEY_FIELDS:
        left = parse_money(predicted, decimal_sep)
        right = parse_money(gold, decimal_sep)
        if left is None or right is None:
            return False
        return left.quantize(Decimal("0.01")) == right.quantize(Decimal("0.01"))

    if field in DATE_FIELDS:
        left, right = normalize_date(predicted), normalize_date(gold)
        return left is not None and left == right

    if field in VENDOR_FIELDS:
        return vendor_matches(predicted, gold, vendor_threshold)

    raise ValueError(f"unknown field: {field!r}")

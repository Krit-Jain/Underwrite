"""Independent validators that flag implausible extractions.

These are black-box signals: they use only the extracted values, never the
model's internals. Each check returns ``1.0`` when it passes and ``0.0`` when it
fails, so the values can be fed straight into the fusion model. A check is
omitted when the fields it needs are absent, rather than being scored as a
failure -- "not applicable" and "wrong" are different facts.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Mapping

from underwrite.data.normalize import MONEY_FIELDS, normalize_date, parse_money
from underwrite.signals.gstin import is_valid_gstin

DEFAULT_TOLERANCE = Decimal("0.01")

# Canonical ordering so callers can build a fixed-width feature vector.
CHECK_NAMES = (
    "valid_money_format",
    "valid_date_format",
    "plausible_date",
    "valid_gstin",
    "sum_matches",
    "tax_split_matches",
)


def _money(fields: Mapping[str, Any], name: str, decimal_sep: str | None) -> Decimal | None:
    if name not in fields:
        return None
    return parse_money(fields.get(name), decimal_sep)


def validate_document(
    fields: Mapping[str, Any],
    *,
    today: date | None = None,
    earliest_date: date = date(2000, 1, 1),
    max_future_days: int = 1,
    tolerance: Decimal = DEFAULT_TOLERANCE,
    decimal_sep: str | None = None,
) -> dict[str, float]:
    """Run every applicable validator over one document's extracted fields.

    ``decimal_sep`` pins the money convention for a corpus (``""`` for rupiah);
    the default infers it per value, which handles both CORD and Indian invoices.
    """
    reference = today or date.today()
    checks: dict[str, float] = {}

    present_money = [name for name in MONEY_FIELDS if fields.get(name) is not None]
    if present_money:
        parsed = {name: _money(fields, name, decimal_sep) for name in present_money}
        checks["valid_money_format"] = float(all(value is not None for value in parsed.values()))

        subtotal, tax, total = parsed.get("subtotal"), parsed.get("tax"), parsed.get("total")
        if None not in (subtotal, tax, total):
            checks["sum_matches"] = float(abs(subtotal + tax - total) <= tolerance)

    if fields.get("date") is not None:
        parsed_date = normalize_date(fields["date"])
        checks["valid_date_format"] = float(parsed_date is not None)
        if parsed_date is not None:
            upper_bound = reference + timedelta(days=max_future_days)
            checks["plausible_date"] = float(earliest_date <= parsed_date <= upper_bound)

    if fields.get("gstin") is not None:
        checks["valid_gstin"] = float(is_valid_gstin(fields["gstin"], require_known_state=True))

    cgst = _money(fields, "cgst", decimal_sep)
    sgst = _money(fields, "sgst", decimal_sep)
    tax = _money(fields, "tax", decimal_sep)
    if None not in (cgst, sgst, tax):
        checks["tax_split_matches"] = float(abs(cgst + sgst - tax) <= tolerance)

    return checks


def as_vector(checks: Mapping[str, float]) -> dict[str, float]:
    """Expand sparse checks to the canonical ordering, defaulting absent to 0.0."""
    return {name: float(checks.get(name, 0.0)) for name in CHECK_NAMES}

"""The canonical field schema and the instruction prompt sent to the vision model."""

from __future__ import annotations

CANONICAL_FIELDS: tuple[str, ...] = ("vendor", "date", "subtotal", "tax", "total")

FIELD_DESCRIPTIONS: dict[str, str] = {
    "vendor": "merchant or seller name",
    "date": "document date in ISO 8601 form (YYYY-MM-DD)",
    "subtotal": "pre-tax amount",
    "tax": "tax amount (GST / VAT / sales tax)",
    "total": "final amount paid",
}

FIELD_TYPES: dict[str, list[str]] = {
    "vendor": ["string", "null"],
    "date": ["string", "null"],
    "subtotal": ["number", "null"],
    "tax": ["number", "null"],
    "total": ["number", "null"],
}

SYSTEM_INSTRUCTION = (
    "You read a single invoice or receipt image and return its key fields as JSON. "
    "Respond with one JSON object and nothing else: no prose and no code fences. "
    "Use null for any field that is absent or unreadable."
)

_VALUE_RULES = (
    "Rules: money fields (subtotal, tax, total) are plain decimals with two places and no "
    "currency symbol or thousands separator; the date is ISO 8601 (YYYY-MM-DD); the vendor "
    "is copied verbatim from the document.\n"
    "Watch the separators. Many receipts -- Indonesian ones especially -- use '.' or ',' "
    "purely as a thousands separator and have no minor unit. On such a document '60.000' "
    "means sixty thousand and must be returned as 60000.00, not as 60.00. A separator "
    "followed by exactly three digits is a thousands group, not a decimal point."
)


def json_schema(fields: tuple[str, ...] = CANONICAL_FIELDS) -> dict[str, object]:
    """JSON Schema describing the extraction target."""
    return {
        "type": "object",
        "properties": {
            name: {"type": FIELD_TYPES[name], "description": FIELD_DESCRIPTIONS[name]}
            for name in fields
        },
        "required": list(fields),
        "additionalProperties": False,
    }


def build_extraction_prompt(fields: tuple[str, ...] = CANONICAL_FIELDS) -> str:
    """Build the user-turn instruction for a single document."""
    lines = ["Extract these fields:", ""]
    lines.extend(f"- {name}: {FIELD_DESCRIPTIONS[name]}" for name in fields)
    lines.extend(["", _VALUE_RULES])
    return "\n".join(lines)

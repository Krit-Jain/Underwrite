"""Recover a JSON object from raw model output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL)


class ParseError(ValueError):
    """Raised when no JSON object can be recovered from the model output."""


def _first_fenced_block(text: str) -> str | None:
    match = _FENCE.search(text)
    return match.group("body") if match else None


def _first_object_span(text: str) -> str | None:
    """Return the first balanced ``{...}`` span, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str | None) -> dict[str, Any]:
    """Return the first JSON object contained in ``text``.

    Handles fenced code blocks and objects embedded in surrounding prose. Raises
    ``ParseError`` when nothing parseable is found, so a bad response is reported
    rather than silently dropped.
    """
    if text is None or not text.strip():
        raise ParseError("model output is empty")

    for candidate in (_first_fenced_block(text), _first_object_span(text), text):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    raise ParseError(f"no JSON object found in output: {text[:200]!r}")

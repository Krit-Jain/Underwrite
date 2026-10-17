import pytest

from underwrite.extract.parse import ParseError, extract_json_object


def test_plain_object():
    assert extract_json_object('{"total": 12.5}') == {"total": 12.5}


def test_fenced_block():
    text = '```json\n{"total": 12.5, "tax": 1.25}\n```'
    assert extract_json_object(text) == {"total": 12.5, "tax": 1.25}


def test_object_embedded_in_prose():
    text = 'Sure, here you go: {"total": 9.99} - hope that helps!'
    assert extract_json_object(text) == {"total": 9.99}


def test_nested_objects_and_braces_inside_strings():
    text = '{"note": "a } brace", "sub": {"x": 1}}'
    assert extract_json_object(text) == {"note": "a } brace", "sub": {"x": 1}}


def test_escaped_quote_inside_string():
    text = r'{"vendor": "Bob\"s Cafe", "total": 5.0}'
    assert extract_json_object(text) == {"vendor": 'Bob"s Cafe', "total": 5.0}


def test_empty_output_raises():
    with pytest.raises(ParseError):
        extract_json_object("   ")


def test_none_output_raises():
    with pytest.raises(ParseError):
        extract_json_object(None)


def test_non_object_raises():
    with pytest.raises(ParseError):
        extract_json_object("[1, 2, 3]")


def test_invalid_json_raises():
    with pytest.raises(ParseError):
        extract_json_object("{not json}")

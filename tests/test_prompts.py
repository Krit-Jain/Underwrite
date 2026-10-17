from underwrite.extract.prompts import (
    CANONICAL_FIELDS,
    FIELD_TYPES,
    build_extraction_prompt,
    json_schema,
)


def test_prompt_names_every_field():
    prompt = build_extraction_prompt()
    for name in CANONICAL_FIELDS:
        assert name in prompt


def test_prompt_can_target_a_subset():
    prompt = build_extraction_prompt(("total",))
    assert "- total:" in prompt
    assert "- vendor:" not in prompt


def test_schema_requires_every_field():
    schema = json_schema()
    assert schema["required"] == list(CANONICAL_FIELDS)
    assert set(schema["properties"]) == set(CANONICAL_FIELDS)


def test_schema_types_match_field_types():
    schema = json_schema()
    for name, types in FIELD_TYPES.items():
        assert schema["properties"][name]["type"] == types


def test_schema_forbids_extra_properties():
    assert json_schema()["additionalProperties"] is False

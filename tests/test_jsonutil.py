import random

import pytest

from neuroclone.llm.jsonutil import JSONExtractError, extract_json, schema_sample, validate


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! ```json\n{"a": [1, 2,],}\n``` hope that helps') == {"a": [1, 2]}
    assert extract_json("prefix {'a': True, 'b': None} suffix") == {"a": True, "b": None}
    assert extract_json('noise {"s": "brace } inside"} more') == {"s": "brace } inside"}
    with pytest.raises(JSONExtractError):
        extract_json("no json here")


def test_validate_reports_path_and_ignores_bad_schemas():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"]}
    assert validate({"n": 2}, schema) is None
    assert "n" in validate({"n": 0}, schema)
    assert "required" in validate({}, schema)
    assert validate({"x": 1}, {"type": "definitely-not-a-type"}) is None
    assert validate({"anything": 1}, None) is None


def test_schema_sample_is_valid_for_common_schemas():
    rng = random.Random(0)
    schemas = [
        {"type": "object", "properties": {"card": {"type": "string", "enum": ["a", "b"]},
                                          "slot": {"type": "integer", "minimum": 1, "maximum": 4}},
         "required": ["card", "slot"]},
        {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer"},
                                                  "minItems": 2, "maxItems": 3, "uniqueItems": True}}},
        {"type": "object", "properties": {"ok": {"type": "boolean"}, "name": {"type": "string", "minLength": 3}},
         "required": ["ok", "name"]},
    ]
    for schema in schemas:
        for _ in range(20):
            sample = schema_sample(schema, rng)
            assert validate(sample, schema) is None, (schema, sample)

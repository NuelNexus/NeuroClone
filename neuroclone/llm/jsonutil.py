"""JSON helpers: tolerant extraction from model output, schema validation, sample generation."""

from __future__ import annotations

import ast
import json
import random
import re
from typing import Any, Optional

import jsonschema
from jsonschema.exceptions import SchemaError


class JSONExtractError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _balanced_span(text: str, start: int) -> Optional[str]:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str: Optional[str] = None
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in "\"'":
            in_str = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1] if ch == closer else None
    return None


def _loads_lenient(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)  # trailing commas
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    pythonish = re.sub(r"\btrue\b", "True", repaired)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish)
    pythonish = re.sub(r"\bnull\b", "None", pythonish)
    try:
        value = ast.literal_eval(pythonish)
    except (ValueError, SyntaxError) as exc:
        raise JSONExtractError(str(exc)) from exc
    if not isinstance(value, (dict, list)):
        raise JSONExtractError("not a JSON object or array")
    return value


def extract_json(text: str) -> Any:
    """Return the first JSON object/array found in ``text`` (fences, prose and trailing commas tolerated)."""
    if text is None:
        raise JSONExtractError("empty response")
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
        for idx, ch in enumerate(chunk):
            if ch in "{[":
                span = _balanced_span(chunk, idx)
                if span is None:
                    continue
                try:
                    return _loads_lenient(span)
                except JSONExtractError:
                    continue
    raise JSONExtractError(f"no JSON found in: {text[:200]!r}")


def validate(instance: Any, schema: Optional[dict]) -> Optional[str]:
    """Return a human-readable error, or None when ``instance`` satisfies ``schema``.

    Invalid schemas (games send all sorts of things) are treated as "anything goes".
    """
    if not schema:
        return None
    try:
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        validator = cls(schema)
    except SchemaError:
        return None
    error = jsonschema.exceptions.best_match(validator.iter_errors(instance))
    if error is None:
        return None
    path = "/".join(str(p) for p in error.absolute_path)
    return f"{path + ': ' if path else ''}{error.message}"


def schema_sample(schema: Optional[dict], rng: Optional[random.Random] = None, depth: int = 0) -> Any:
    """Generate a plausible instance of ``schema`` (like the SDK's Randy test bot)."""
    rng = rng or random.Random()
    if not schema or depth > 6:
        return {} if depth == 0 else None
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return rng.choice(schema["enum"])
    for key in ("anyOf", "oneOf"):
        if schema.get(key):
            return schema_sample(rng.choice(schema[key]), rng, depth + 1)
    if "default" in schema:
        return schema["default"]
    if schema.get("examples"):
        return rng.choice(schema["examples"])
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), typ[0] if typ else None)
    if typ is None:
        typ = "object" if "properties" in schema else "string"
    if typ == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        out = {}
        for name, sub in props.items():
            if name in required or rng.random() < 0.5:
                out[name] = schema_sample(sub, rng, depth + 1)
        return out
    if typ == "array":
        lo = int(schema.get("minItems", 1))
        hi = int(schema.get("maxItems", max(lo, 3)))
        count = rng.randint(lo, max(lo, hi))
        items = schema.get("items", {"type": "string"})
        values = [schema_sample(items, rng, depth + 1) for _ in range(count)]
        if schema.get("uniqueItems"):
            seen, unique = set(), []
            for v in values:
                key = json.dumps(v, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    unique.append(v)
            values = unique
        return values
    if typ in ("integer", "number"):
        lo = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        hi = schema.get("maximum", schema.get("exclusiveMaximum", lo + 10))
        if "exclusiveMinimum" in schema and "minimum" not in schema:
            lo = lo + (1 if typ == "integer" else 0.001)
        if "exclusiveMaximum" in schema and "maximum" not in schema:
            hi = hi - (1 if typ == "integer" else 0.001)
        if hi < lo:
            hi = lo
        return rng.randint(int(lo), int(hi)) if typ == "integer" else round(rng.uniform(lo, hi), 3)
    if typ == "boolean":
        return rng.random() < 0.5
    if typ == "null":
        return None
    lo = int(schema.get("minLength", 1))
    hi = int(schema.get("maxLength", max(lo, 12)))
    fmt = schema.get("format")
    if fmt == "date":
        return "2026-01-01"
    if fmt == "date-time":
        return "2026-01-01T00:00:00Z"
    word = "neuro"
    text = (word * (lo // len(word) + 1))[: max(lo, min(hi, len(word)))]
    return text

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import jsonschema
from jsonschema import Draft202012Validator


class SchemaValidationError(ValueError):
    """Raised when JSON schema validation fails."""

    def __init__(self, schema_name: str, message: str) -> None:
        super().__init__(f"Schema '{schema_name}' validation error: {message}")
        self.schema_name = schema_name
        self.detail = message


def _format_path(path: Iterable[object]) -> str:
    formatted = "$"
    for part in path:
        if isinstance(part, int):
            formatted += f"[{part}]"
        else:
            formatted += f".{part}"
    return formatted


def load_schema(schema_name: str, schemas_dir: Path | None = None) -> dict[str, Any]:
    if schemas_dir is None:
        # __file__ is src/scenarioops/graph/tools/schema_validate.py
        # parents[2] is scenarioops
        schemas_dir = Path(__file__).resolve().parents[2] / "schemas"
    schema_path = schemas_dir / f"{schema_name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if "$id" not in schema:
        schema["$id"] = schema_path.resolve().as_uri()
    return schema


def validate_schema(instance: Any, schema: dict[str, Any], schema_name: str) -> None:
    validator = Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: _format_path(error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = _format_path(error.absolute_path)
    raise SchemaValidationError(schema_name, f"{path}: {error.message}")


def validate_artifact(
    schema_name: str, instance: Any, schemas_dir: Path | None = None
) -> None:
    schema = load_schema(schema_name, schemas_dir=schemas_dir)
    validate_schema(instance, schema, schema_name)


def validate_jsonl(
    schema_name: str, items: Iterable[Any], schemas_dir: Path | None = None
) -> None:
    schema = load_schema(schema_name, schemas_dir=schemas_dir)
    for item in items:
        validate_schema(item, schema, schema_name)

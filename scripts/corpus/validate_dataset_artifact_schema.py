"""Validate dataset JSONL rows against project artifact schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_row(row: dict, schema: dict) -> list[str]:
    errors: list[str] = []
    required = schema.get("required", [])
    props = schema.get("properties", {})
    for key in required:
        if key not in row:
            errors.append(f"missing required field '{key}'")
    for key, rules in props.items():
        if key not in row:
            continue
        value = row[key]
        expected_type = rules.get("type")
        if expected_type == "string":
            if not isinstance(value, str):
                errors.append(f"field '{key}' expected string")
                continue
            min_len = int(rules.get("minLength", 0))
            if len(value) < min_len:
                errors.append(f"field '{key}' too short")
        enum = rules.get("enum")
        if enum and value not in enum:
            errors.append(f"field '{key}' not in enum {enum}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--schema", default="config/dataset_artifact_schema.json")
    args = parser.parse_args()

    schema = _load_schema(Path(args.schema))
    lines = Path(args.input_jsonl).read_text(encoding="utf-8").splitlines()
    failed = 0
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        errors = _validate_row(row, schema)
        if errors:
            failed += 1
            print(f"line={idx} errors={'; '.join(errors)}")
    if failed:
        print(f"FAILED rows={failed}")
        return 2
    print("OK schema valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

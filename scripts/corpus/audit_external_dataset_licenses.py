"""Audit dataset licenses and block incompatible updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "cc-by-4.0",
}


def _normalize(value: str) -> str:
    return value.strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=".cursor/artifacts/quality/dataset_manifest.json"
    )
    parser.add_argument(
        "--previous-manifest",
        default=".cursor/artifacts/quality/dataset_manifest.last_good.json",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_payload = {}
    previous_path = Path(args.previous_manifest)
    if previous_path.is_file():
        previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))

    previous_by_id = {row["id"]: row for row in previous_payload.get("sources", [])}
    errors: list[str] = []
    warnings: list[str] = []
    for row in payload.get("sources", []):
        source_id = str(row.get("id", ""))
        license_name = _normalize(str(row.get("license", "")))
        if license_name not in ALLOWED_LICENSES:
            errors.append(f"{source_id}: license '{license_name}' is not allowed")
        prev = previous_by_id.get(source_id)
        if prev:
            prev_license = _normalize(str(prev.get("license", "")))
            if prev_license != license_name:
                errors.append(
                    f"{source_id}: license changed {prev_license!r} -> {license_name!r}, update blocked"
                )
        if not bool(row.get("download_ok", False)):
            warnings.append(
                f"{source_id}: source unavailable, using last-known-good if present"
            )

    if warnings:
        for warning in warnings:
            print(f"WARNING {warning}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 2
    print("OK licenses validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

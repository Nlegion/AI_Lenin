#!/usr/bin/env python3
"""Create a hash manifest for subplan reproducibility artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield child


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build reproducibility manifest for subplan inputs/outputs."
    )
    parser.add_argument(
        "--subplan", required=True, help="Subplan identifier, for example 'A'."
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="File or directory to hash (repeatable).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output manifest path, for example .cursor/artifacts/manifests/20260718-subplan-a.json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    files_payload: list[dict[str, str | int]] = []

    for raw_path in args.path:
        absolute_path = (repo_root / raw_path).resolve()
        if not absolute_path.exists():
            continue
        for file_path in _iter_files(path=absolute_path):
            relative_path = file_path.relative_to(repo_root).as_posix()
            files_payload.append(
                {
                    "path": relative_path,
                    "sha256": _sha256(file_path=file_path),
                    "size_bytes": file_path.stat().st_size,
                }
            )

    output_path = (repo_root / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subplan": args.subplan,
        "generated_at_utc": timestamp,
        "files": files_payload,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Manifest written to {output_path}")
    print(f"Tracked files: {len(files_payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

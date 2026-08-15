"""Mandatory external dataset pre-step for censorship quality/replay scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_external_dataset_prestep(*, repo_root: Path, max_rows_per_source: int) -> None:
    scripts_dir = repo_root / "scripts"
    materialize_script = scripts_dir / "materialize_external_datasets.py"
    validate_script = scripts_dir / "validate_dataset_artifact_schema.py"
    unified_jsonl = repo_root / "data" / "external_datasets" / "external_unified.jsonl"
    schema_path = repo_root / "config" / "dataset_artifact_schema.json"

    subprocess.run(  # nosec B603
        [
            sys.executable,
            str(materialize_script),
            "--max-rows-per-source",
            str(max_rows_per_source),
        ],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(  # nosec B603
        [
            sys.executable,
            str(validate_script),
            "--input-jsonl",
            str(unified_jsonl),
            "--schema",
            str(schema_path),
        ],
        cwd=repo_root,
        check=True,
    )


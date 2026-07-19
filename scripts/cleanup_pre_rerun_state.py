#!/usr/bin/env python3
"""Clean stale artifacts before full E->J rerun."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path


def _remove_path(path: Path) -> dict[str, str | int]:
    if not path.exists():
        return {"path": path.as_posix(), "status": "missing", "removed_files": 0}

    removed_files = 0
    if path.is_file():
        path.unlink()
        removed_files = 1
        return {"path": path.as_posix(), "status": "removed_file", "removed_files": removed_files}

    for file_path in sorted(path.rglob("*"), reverse=True):
        if file_path.is_file():
            file_path.unlink()
            removed_files += 1
    for dir_path in sorted(path.rglob("*"), reverse=True):
        if dir_path.is_dir():
            dir_path.rmdir()
    path.rmdir()
    return {"path": path.as_posix(), "status": "removed_dir", "removed_files": removed_files}


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    targets = [
        repo / "database/qdrant_local",
        repo / ".cursor/artifacts/chunks/chunk_dataset_v2.tsv",
        repo / ".cursor/artifacts/qdrant/checkpoints",
        repo / ".cursor/artifacts/retrieval/retrieval_ab_audit.jsonl",
        repo / ".cursor/artifacts/retrieval/retrieval_ab_summary.md",
        repo / ".cursor/artifacts/retrieval/retrieval_ab_summary.json",
        repo / ".cursor/artifacts/evaluation/rag_quality_metrics.json",
        repo / ".cursor/artifacts/evaluation/rag_quality_summary.md",
        repo / ".cursor/artifacts/cleaning/cleaning_qa.json",
        repo / ".cursor/artifacts/cleaning/cleaning_summary.md",
    ]

    results = [_remove_path(path=target) for target in targets]
    report = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }

    output = repo / ".cursor/artifacts/20260718-2125-pre-rerun-cleanup.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = repo / ".cursor/artifacts/20260718-2125-pre-rerun-cleanup.md"
    lines = [
        "# Pre-rerun Cleanup Report",
        "",
        f"- Generated at (UTC): `{report['generated_at_utc']}`",
        "",
        "| Target | Status | Removed files |",
        "|---|---|---:|",
    ]
    for row in results:
        lines.append(f"| `{row['path']}` | `{row['status']}` | {row['removed_files']} |")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"cleanup_targets {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

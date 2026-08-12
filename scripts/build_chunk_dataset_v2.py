#!/usr/bin/env python3
"""Build chunk dataset v2 from cleaned corpus and source registry."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.preprocessing.chunker_v2 import chunk_document  # noqa: E402
from src.core.preprocessing.chunking_config import load_chunking_config  # noqa: E402
from src.core.preprocessing.chunking_quality import bad_boundary_ratio, token_window_compliance_ratio  # noqa: E402


def _read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _write_chunks_tsv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Chunking V2 Summary",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Source documents processed: `{summary['source_documents_processed']}`",
        f"- Total chunks: `{summary['total_chunks']}`",
        f"- Mean tokens per chunk: `{summary['mean_tokens']:.2f}`",
        f"- Token window compliance ratio: `{summary['token_window_compliance_ratio']:.4f}`",
        f"- Bad boundary ratio: `{summary['bad_boundary_ratio']:.4f}`",
        f"- Max bad boundary ratio target: `{summary['max_bad_boundary_ratio_target']:.4f}`",
        f"- Boundary target passed: `{'yes' if summary['boundary_target_passed'] else 'no'}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build chunk dataset v2.")
    parser.add_argument("--registry", default=".cursor/artifacts/registries/source_registry.tsv")
    parser.add_argument("--cleaned-root", default=".cursor/artifacts/cleaned_corpus")
    parser.add_argument("--config", default="config/chunking_rules.yaml")
    parser.add_argument("--chunks-output", default=".cursor/artifacts/chunks/chunk_dataset_v2.tsv")
    parser.add_argument("--summary-output", default=".cursor/artifacts/chunks/chunking_summary.md")
    parser.add_argument("--qa-output", default=".cursor/artifacts/chunks/chunking_qa.json")
    args = parser.parse_args()

    registry_path = (REPO_ROOT / args.registry).resolve()
    cleaned_root = (REPO_ROOT / args.cleaned_root).resolve()
    config_path = (REPO_ROOT / args.config).resolve()
    chunks_output = (REPO_ROOT / args.chunks_output).resolve()
    summary_output = (REPO_ROOT / args.summary_output).resolve()
    qa_output = (REPO_ROOT / args.qa_output).resolve()

    config = load_chunking_config(path=config_path)
    rows = _read_registry(path=registry_path)
    all_chunks = []

    for row in rows:
        source_path = row["source_path"]
        text_path = cleaned_root / source_path
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_document(
            source_id=row["source_id"],
            source_path=source_path,
            author=row["author"],
            work=row["work"],
            stance_type=row["stance_type"],
            text=text,
            config=config,
        )
        all_chunks.extend(chunks)

    chunk_rows = [
        {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "source_path": chunk.source_path,
            "author": chunk.author,
            "work": chunk.work,
            "stance_type": chunk.stance_type,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "paragraph_index": chunk.paragraph_index,
            "thesis_index": chunk.thesis_index,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "boundary_ok": chunk.boundary_ok,
            "text": chunk.text,
        }
        for chunk in all_chunks
    ]
    _write_chunks_tsv(rows=chunk_rows, output_path=chunks_output)

    token_counts = [chunk.token_count for chunk in all_chunks]
    mean_tokens = (sum(token_counts) / len(token_counts)) if token_counts else 0.0
    boundary_ratio = bad_boundary_ratio(chunks=all_chunks)
    compliance_ratio = token_window_compliance_ratio(
        chunks=all_chunks,
        min_tokens=config.min_tokens,
        max_tokens=config.max_tokens,
    )
    summary = {
        "source_documents_processed": len(rows),
        "total_chunks": len(all_chunks),
        "mean_tokens": mean_tokens,
        "token_window_compliance_ratio": compliance_ratio,
        "bad_boundary_ratio": boundary_ratio,
        "max_bad_boundary_ratio_target": config.max_bad_boundary_ratio,
        "boundary_target_passed": boundary_ratio <= config.max_bad_boundary_ratio,
    }

    qa_output.parent.mkdir(parents=True, exist_ok=True)
    qa_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary=summary, output_path=summary_output)

    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Boundary target passed: {summary['boundary_target_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build deterministic retrieval eval dataset from ontology tags."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle, delimiter="\t")
        return list(reader)


def _query_for_concept(concept: str) -> str:
    return f"Instruct: Given a question, retrieve passages that answer the question\nQuery: {concept}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build embedding eval dataset from ontology tags.")
    parser.add_argument(
        "--ontology-tags",
        default=".cursor/artifacts/ontology/ontology_tags.tsv",
        help="Input ontology tags TSV.",
    )
    parser.add_argument(
        "--output",
        default=".cursor/artifacts/eval/embedding_eval.tsv",
        help="Output eval TSV.",
    )
    parser.add_argument("--max-rows", type=int, default=120, help="Maximum number of eval rows.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    tags_path = (repo_root / args.ontology_tags).resolve()
    output_path = (repo_root / args.output).resolve()

    rows = _read_rows(path=tags_path)
    candidates: list[dict[str, str]] = []
    for row in rows:
        concepts = [item for item in row.get("concepts", "").split("|") if item]
        if not concepts:
            continue
        primary_concept = concepts[0]
        candidates.append(
            {
                "query": _query_for_concept(primary_concept),
                "positive_source_id": row["source_id"],
                "positive_source_path": row["source_path"],
                "concept": primary_concept,
            }
        )

    random.seed(42)
    random.shuffle(candidates)
    selected = candidates[: args.max_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        fieldnames = ["query", "positive_source_id", "positive_source_path", "concept"]
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in selected:
            writer.writerow(row)

    print(f"Eval dataset written: {output_path}")
    print(f"Rows: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

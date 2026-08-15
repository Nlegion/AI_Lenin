#!/usr/bin/env python3
"""Audit ontology tags and chunk stance distribution after full rerun."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "unknown") or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit retrieval foundations.")
    parser.add_argument(
        "--registry", default=".cursor/artifacts/registries/source_registry.tsv"
    )
    parser.add_argument(
        "--ontology-tags", default=".cursor/artifacts/ontology/ontology_tags.tsv"
    )
    parser.add_argument(
        "--chunks", default=".cursor/artifacts/chunks/chunk_dataset_v2.tsv"
    )
    parser.add_argument(
        "--out-json",
        default=".cursor/artifacts/evaluation/retrieval_foundations_audit.json",
    )
    parser.add_argument(
        "--out-md",
        default=".cursor/artifacts/evaluation/retrieval_foundations_audit.md",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    registry_rows = _read_tsv(path=(repo / args.registry).resolve())
    ontology_rows = _read_tsv(path=(repo / args.ontology_tags).resolve())
    chunk_rows = _read_tsv(path=(repo / args.chunks).resolve())

    registry_stance = _count_by(rows=registry_rows, key="stance_type")
    ontology_stance = _count_by(rows=ontology_rows, key="stance_type")
    chunk_stance = _count_by(rows=chunk_rows, key="stance_type")
    ontology_sources = {row["source_id"] for row in ontology_rows}
    registry_sources = {row["source_id"] for row in registry_rows}
    missing_in_ontology = sorted(registry_sources - ontology_sources)

    payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registry_documents": len(registry_rows),
        "ontology_documents": len(ontology_rows),
        "chunk_rows": len(chunk_rows),
        "registry_stance_distribution": registry_stance,
        "ontology_stance_distribution": ontology_stance,
        "chunk_stance_distribution": chunk_stance,
        "missing_registry_sources_in_ontology": missing_in_ontology,
    }

    out_json = (repo / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out_md = (repo / args.out_md).resolve()
    lines = [
        "# Retrieval Foundations Audit",
        "",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        f"- Registry documents: `{payload['registry_documents']}`",
        f"- Ontology documents: `{payload['ontology_documents']}`",
        f"- Chunk rows: `{payload['chunk_rows']}`",
        f"- Missing registry sources in ontology: `{len(missing_in_ontology)}`",
        "",
        "## Stance Distribution",
        "",
        "| Layer | core_self | influence_agree | influence_critical | contextual |",
        "|---|---:|---:|---:|---:|",
        (
            f"| registry | {registry_stance.get('core_self', 0)} | "
            f"{registry_stance.get('influence_agree', 0)} | "
            f"{registry_stance.get('influence_critical', 0)} | "
            f"{registry_stance.get('contextual', 0)} |"
        ),
        (
            f"| ontology_tags | {ontology_stance.get('core_self', 0)} | "
            f"{ontology_stance.get('influence_agree', 0)} | "
            f"{ontology_stance.get('influence_critical', 0)} | "
            f"{ontology_stance.get('contextual', 0)} |"
        ),
        (
            f"| chunks | {chunk_stance.get('core_self', 0)} | "
            f"{chunk_stance.get('influence_agree', 0)} | "
            f"{chunk_stance.get('influence_critical', 0)} | "
            f"{chunk_stance.get('contextual', 0)} |"
        ),
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"missing_in_ontology {len(missing_in_ontology)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

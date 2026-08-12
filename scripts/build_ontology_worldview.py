#!/usr/bin/env python3
"""Build ontology tags and worldview graph from source registry."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.ontology.tagger import tag_document  # noqa: E402
from src.core.ontology.taxonomy import load_taxonomy  # noqa: E402
from src.core.ontology.worldview_graph import TaggedSource, build_worldview_graph  # noqa: E402


def _load_registry(registry_path: Path) -> list[dict[str, str]]:
    with registry_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle, delimiter="\t")
        return list(reader)


def _read_text(corpus_root: Path, source_path: str) -> str:
    file_path = corpus_root / source_path
    return file_path.read_text(encoding="utf-8", errors="replace")


def _write_tags_tsv(records: list[TaggedSource], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "source_path",
        "stance_type",
        "concepts",
        "entities",
        "contradiction_hits",
        "argument_pattern",
        "zero_shot_label",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_id": record.source_id,
                    "source_path": record.source_path,
                    "stance_type": record.stance_type,
                    "concepts": "|".join(record.concepts),
                    "entities": "|".join(record.entities),
                    "contradiction_hits": "|".join(record.contradiction_hits),
                    "argument_pattern": record.argument_pattern,
                    "zero_shot_label": record.zero_shot_label,
                }
            )


def _write_validation_sample(records: list[TaggedSource], output_path: Path, sample_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    sampled = random.sample(records, min(sample_size, len(records)))
    fieldnames = [
        "source_id",
        "source_path",
        "auto_zero_shot_label",
        "annotator_a",
        "annotator_b",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in sampled:
            writer.writerow(
                {
                    "source_id": record.source_id,
                    "source_path": record.source_path,
                    "auto_zero_shot_label": record.zero_shot_label,
                    "annotator_a": "",
                    "annotator_b": "",
                    "notes": "",
                }
            )


def _compute_iaa(validation_path: Path) -> float | None:
    with validation_path.open("r", encoding="utf-8", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle, delimiter="\t"))
    filtered = [
        row for row in rows if row.get("annotator_a", "").strip() and row.get("annotator_b", "").strip()
    ]
    if not filtered:
        return None
    total = len(filtered)
    agreement = sum(1 for row in filtered if row["annotator_a"].strip() == row["annotator_b"].strip())
    return agreement / total


def _write_summary(
    tagged_records: list[TaggedSource],
    graph_payload: dict[str, list[dict[str, str | int]]],
    summary_path: Path,
    iaa_score: float | None,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    contradiction_docs = sum(1 for record in tagged_records if record.contradiction_hits)
    lines = [
        "# Ontology Stage Summary",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Tagged documents: `{len(tagged_records)}`",
        f"- Graph nodes: `{len(graph_payload['nodes'])}`",
        f"- Graph edges: `{len(graph_payload['edges'])}`",
        f"- Documents with contradiction hits: `{contradiction_docs}`",
    ]
    if iaa_score is None:
        lines.append("- IAA (annotator_a vs annotator_b): `N/A` (manual annotations not filled yet)")
    else:
        lines.append(f"- IAA (annotator_a vs annotator_b): `{iaa_score:.3f}`")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ontology tags and worldview graph.")
    parser.add_argument(
        "--registry",
        default=".cursor/artifacts/registries/source_registry.tsv",
        help="Input source registry TSV path.",
    )
    parser.add_argument("--corpus-root", default="data/books", help="Corpus root path.")
    parser.add_argument(
        "--taxonomy-config",
        default="config/ontology_taxonomy.yaml",
        help="Ontology taxonomy YAML path.",
    )
    parser.add_argument(
        "--tags-output",
        default=".cursor/artifacts/ontology/ontology_tags.tsv",
        help="Output TSV for document tags.",
    )
    parser.add_argument(
        "--graph-output",
        default=".cursor/artifacts/ontology/worldview_graph.json",
        help="Output JSON worldview graph.",
    )
    parser.add_argument(
        "--validation-output",
        default=".cursor/artifacts/ontology/validation_sample.tsv",
        help="Output TSV for manual annotation sample.",
    )
    parser.add_argument(
        "--summary-output",
        default=".cursor/artifacts/ontology/ontology_summary.md",
        help="Output summary markdown path.",
    )
    parser.add_argument("--sample-size", type=int, default=25, help="Validation sample size.")
    args = parser.parse_args()

    registry_path = (REPO_ROOT / args.registry).resolve()
    corpus_root = (REPO_ROOT / args.corpus_root).resolve()
    taxonomy_config = (REPO_ROOT / args.taxonomy_config).resolve()
    tags_output = (REPO_ROOT / args.tags_output).resolve()
    graph_output = (REPO_ROOT / args.graph_output).resolve()
    validation_output = (REPO_ROOT / args.validation_output).resolve()
    summary_output = (REPO_ROOT / args.summary_output).resolve()

    taxonomy = load_taxonomy(config_path=taxonomy_config)
    registry_rows = _load_registry(registry_path=registry_path)
    tagged_sources: list[TaggedSource] = []
    for row in registry_rows:
        text = _read_text(corpus_root=corpus_root, source_path=row["source_path"])
        tags = tag_document(text=text, taxonomy=taxonomy)
        tagged_sources.append(
            TaggedSource(
                source_id=row["source_id"],
                source_path=row["source_path"],
                stance_type=row["stance_type"],
                concepts=tags.concepts,
                entities=tags.entities,
                contradiction_hits=tags.contradiction_hits,
                argument_pattern=tags.argument_pattern,
                zero_shot_label=tags.zero_shot_label,
            )
        )

    _write_tags_tsv(records=tagged_sources, output_path=tags_output)
    graph_payload = build_worldview_graph(tagged_sources=tagged_sources)
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    graph_output.write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_validation_sample(records=tagged_sources, output_path=validation_output, sample_size=args.sample_size)
    iaa_score = _compute_iaa(validation_path=validation_output)
    _write_summary(
        tagged_records=tagged_sources,
        graph_payload=graph_payload,
        summary_path=summary_output,
        iaa_score=iaa_score,
    )

    print(f"Tagged records: {len(tagged_sources)}")
    print(f"Graph nodes: {len(graph_payload['nodes'])}")
    print(f"Graph edges: {len(graph_payload['edges'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

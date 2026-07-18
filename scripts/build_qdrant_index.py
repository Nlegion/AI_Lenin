#!/usr/bin/env python3
"""Build Qdrant local index from chunk dataset v2."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.vector.qdrant_ingestion import IngestionConfig, QdrantIngestionPipeline  # noqa: E402


def _load_config(path: Path) -> IngestionConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("qdrant_ingestion", payload)
    return IngestionConfig(
        collection_name=section["collection_name"],
        dense_model=section["dense_model"],
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        device=section.get("device", "cpu"),
        batch_size=int(section.get("batch_size", 64)),
        retries=int(section.get("retries", 2)),
        checkpoint_path=(REPO_ROOT / section["checkpoint_path"]).resolve(),
        qdrant_path=(REPO_ROOT / section["qdrant_path"]).resolve(),
        sparse_state_path=(REPO_ROOT / section["sparse_state_path"]).resolve(),
        prewarm_core_limit=int(section.get("prewarm_core_limit", 200)),
    )


def _write_summary(path: Path, stats: dict[str, int | float], config: IngestionConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qdrant Ingestion Summary",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Collection: `{config.collection_name}`",
        f"- Dense model: `{config.dense_model}`",
        f"- Rows total: `{stats['rows_total']}`",
        f"- Rows processed this run: `{stats['rows_processed']}`",
        f"- Checkpoint offset: `{stats['checkpoint_offset']}`",
        f"- Prewarmed points: `{stats['prewarmed_points']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Qdrant local index from chunks.")
    parser.add_argument("--config", default="config/qdrant_ingestion.yaml")
    parser.add_argument("--chunks-tsv", default=".cursor/artifacts/chunks/chunk_dataset_v2.tsv")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke runs.")
    parser.add_argument("--stats-json", default=".cursor/artifacts/qdrant/ingestion_stats.json")
    parser.add_argument("--summary-md", default=".cursor/artifacts/qdrant/ingestion_summary.md")
    args = parser.parse_args()

    config = _load_config(path=(REPO_ROOT / args.config).resolve())
    pipeline = QdrantIngestionPipeline(config=config)
    stats = pipeline.run(
        chunks_tsv_path=(REPO_ROOT / args.chunks_tsv).resolve(),
        limit=args.limit if args.limit > 0 else None,
    )

    stats_path = (REPO_ROOT / args.stats_json).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(path=(REPO_ROOT / args.summary_md).resolve(), stats=stats, config=config)

    print(f"Rows processed: {stats['rows_processed']}")
    print(f"Prewarmed points: {stats['prewarmed_points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

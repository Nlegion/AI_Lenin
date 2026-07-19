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

from src.core.settings.device import GIGA_EMBEDDING_DIM, hardware_report, resolve_torch_device  # noqa: E402
from src.core.vector.qdrant_ingestion import IngestionConfig, QdrantIngestionPipeline  # noqa: E402


def _resolve_dense_model(model_name: str) -> tuple[str, Path | None]:
    candidate = Path(model_name)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if candidate.exists():
        return str(candidate.resolve()), candidate.resolve()
    return model_name, None


def _load_config(path: Path, *, reset_checkpoint: bool) -> IngestionConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("qdrant_ingestion", payload)
    dense_model, model_dir = _resolve_dense_model(model_name=section["dense_model"])
    return IngestionConfig(
        collection_name=section["collection_name"],
        dense_model=dense_model,
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        device=str(section.get("device", "auto")),
        batch_size=int(section.get("batch_size", 16)),
        retries=int(section.get("retries", 2)),
        checkpoint_path=(REPO_ROOT / section["checkpoint_path"]).resolve(),
        qdrant_path=(REPO_ROOT / section["qdrant_path"]).resolve(),
        sparse_state_path=(REPO_ROOT / section["sparse_state_path"]).resolve(),
        prewarm_core_limit=int(section.get("prewarm_core_limit", 200)),
        fallback_to_cpu=bool(section.get("fallback_to_cpu", True)),
        adaptive_batch=bool(section.get("adaptive_batch", True)),
        min_batch_size=int(section.get("min_batch_size", 4)),
        expected_dim=int(section.get("expected_dim", GIGA_EMBEDDING_DIM)),
        model_dir=model_dir,
        reset_checkpoint=reset_checkpoint,
    )


def _write_summary(path: Path, stats: dict, config: IngestionConfig, hw: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qdrant Ingestion Summary",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Collection: `{config.collection_name}`",
        f"- Dense model: `{config.dense_model}`",
        f"- Device preference: `{config.device}` resolved=`{hw.get('resolved_device')}`",
        f"- Torch: `{hw.get('torch_version')}` GPU=`{hw.get('gpu_name')}`",
        f"- Rows total: `{stats['rows_total']}`",
        f"- Rows processed this run: `{stats['rows_processed']}`",
        f"- Checkpoint offset: `{stats['checkpoint_offset']}`",
        f"- Final batch size: `{stats.get('final_batch_size', config.batch_size)}`",
        f"- Prewarmed points: `{stats['prewarmed_points']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Qdrant local index from chunks.")
    parser.add_argument("--config", default="config/qdrant_ingestion.yaml")
    parser.add_argument("--chunks-tsv", default=".cursor/artifacts/chunks/chunk_dataset_v2.tsv")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke runs.")
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--stats-json", default=".cursor/artifacts/qdrant/ingestion_stats_giga_v1.json")
    parser.add_argument("--summary-md", default=".cursor/artifacts/qdrant/ingestion_summary_giga_v1.md")
    args = parser.parse_args()

    config = _load_config(
        path=(REPO_ROOT / args.config).resolve(),
        reset_checkpoint=args.reset_checkpoint,
    )
    resolved = resolve_torch_device(
        preferred=config.device,
        fallback_to_cpu=config.fallback_to_cpu,
    )
    hw = hardware_report(resolved_device=resolved, fallback_to_cpu=config.fallback_to_cpu)
    pipeline = QdrantIngestionPipeline(config=config)
    try:
        stats = pipeline.run(
            chunks_tsv_path=(REPO_ROOT / args.chunks_tsv).resolve(),
            limit=args.limit if args.limit > 0 else None,
        )
    finally:
        pipeline.close()

    stats_path = (REPO_ROOT / args.stats_json).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stats": stats, "hardware": hw}
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(
        path=(REPO_ROOT / args.summary_md).resolve(),
        stats=stats,
        config=config,
        hw=hw,
    )
    print(f"Rows processed: {stats['rows_processed']}")
    print(f"Checkpoint: {stats['checkpoint_offset']}")
    print(f"Device: {hw['resolved_device']} torch={hw['torch_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify and optionally pack the local RAG snapshot for VPS Docker deploy."""

from __future__ import annotations

import argparse
import logging
import tarfile
from pathlib import Path

from src.core.retrieval.provider_factory import load_retrieval_pipeline_config
from src.core.retrieval.rag_preflight import (
    RagPreflightError,
    dir_size_bytes,
    run_rag_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("pack_rag_snapshot")


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num} B"


def collect_snapshot_paths(base_dir: Path) -> dict[str, Path]:
    config = load_retrieval_pipeline_config(
        config_path=base_dir / "config" / "retrieval_pipeline.yaml"
    )
    return {
        "dense_model": base_dir / config.dense_model,
        "qdrant_path": base_dir / config.qdrant_path,
        "sparse_state_path": base_dir / config.sparse_state_path,
        "ontology_tags_path": base_dir / config.ontology_tags_path,
    }


def verify_snapshot(base_dir: Path) -> dict[str, Path]:
    run_rag_preflight(base_dir=base_dir)
    paths = collect_snapshot_paths(base_dir=base_dir)
    for label, path in paths.items():
        size = dir_size_bytes(path) if path.is_dir() else path.stat().st_size
        logger.info(
            "snapshot_asset label=%s path=%s size=%s",
            label,
            path,
            _human_bytes(size),
        )
    return paths


def write_tarball(*, base_dir: Path, output: Path, paths: dict[str, Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w:gz") as archive:
        for path in paths.values():
            arcname = path.relative_to(base_dir).as_posix()
            archive.add(path, arcname=arcname)
    logger.info(
        "wrote_tarball path=%s size=%s", output, _human_bytes(output.stat().st_size)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify (and optionally tar) the local RAG snapshot for Docker VPS."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: project root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .tar.gz path; when set, packs verified assets",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        paths = verify_snapshot(base_dir=base_dir)
    except RagPreflightError as error:
        logger.error("snapshot_verify_failed detail=%s", error)
        return 1
    if args.output is not None:
        write_tarball(base_dir=base_dir, output=args.output.resolve(), paths=paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

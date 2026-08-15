"""Artifact path helpers for live-news QA batch."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scripts.lib._quality_qa_io import ArtifactPaths


def resolve_live_artifacts(
    *,
    output_dir: Path,
    stem: str,
    checkpoint: Path | None,
) -> ArtifactPaths:
    if checkpoint is not None:
        ckpt = checkpoint
        name = ckpt.name
        if name.endswith(".checkpoint.jsonl"):
            base_stem = name[: -len(".checkpoint.jsonl")]
            parent = ckpt.parent
            return ArtifactPaths(
                checkpoint=ckpt,
                results=parent / f"{base_stem}.jsonl",
                txt=parent / f"{base_stem}.txt",
            )
        return ArtifactPaths(
            checkpoint=ckpt,
            results=ckpt.with_name(f"{ckpt.name}.results.jsonl"),
            txt=ckpt.with_name(f"{ckpt.name}.txt"),
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    full = f"{stem}_{stamp}"
    return ArtifactPaths(
        checkpoint=output_dir / f"{full}.checkpoint.jsonl",
        results=output_dir / f"{full}.jsonl",
        txt=output_dir / f"{full}.txt",
    )


def rejected_paths(artifacts: ArtifactPaths) -> tuple[Path, Path]:
    stem = artifacts.results.name
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    parent = artifacts.results.parent
    return parent / f"{stem}.rejected.jsonl", parent / f"{stem}.rejected.txt"


def count_done_rows(prior: dict) -> int:
    return sum(1 for row in prior.values() if str(row.get("status")) == "done")

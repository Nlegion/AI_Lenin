"""Configuration for chunking v2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ChunkingConfig:
    min_tokens: int
    max_tokens: int
    overlap_ratio: float
    thesis_markers: list[str]
    chapter_markers: list[str]
    section_markers: list[str]
    min_chunk_chars: int
    max_bad_boundary_ratio: float


def load_chunking_config(path: Path) -> ChunkingConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("chunking", payload)
    return ChunkingConfig(
        min_tokens=int(section.get("min_tokens", 256)),
        max_tokens=int(section.get("max_tokens", 512)),
        overlap_ratio=float(section.get("overlap_ratio", 0.1)),
        thesis_markers=list(section.get("thesis_markers", [])),
        chapter_markers=list(section.get("chapter_markers", [])),
        section_markers=list(section.get("section_markers", [])),
        min_chunk_chars=int(section.get("min_chunk_chars", 240)),
        max_bad_boundary_ratio=float(section.get("max_bad_boundary_ratio", 0.2)),
    )

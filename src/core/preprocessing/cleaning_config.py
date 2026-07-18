"""Configuration loading for corpus cleaning stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CleaningConfig:
    remove_line_patterns: list[str]
    remove_inline_patterns: list[str]
    content_start_markers: list[str]
    min_cleaned_chars: int
    min_semantic_paragraph_chars: int
    semantic_overlap_threshold: float
    validation_sample_size: int
    max_semantic_damage_ratio: float


def load_cleaning_config(path: Path) -> CleaningConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("cleaning", payload)
    return CleaningConfig(
        remove_line_patterns=list(section.get("remove_line_patterns", [])),
        remove_inline_patterns=list(section.get("remove_inline_patterns", [])),
        content_start_markers=list(section.get("content_start_markers", [])),
        min_cleaned_chars=int(section.get("min_cleaned_chars", 600)),
        min_semantic_paragraph_chars=int(section.get("min_semantic_paragraph_chars", 120)),
        semantic_overlap_threshold=float(section.get("semantic_overlap_threshold", 0.4)),
        validation_sample_size=int(section.get("validation_sample_size", 25)),
        max_semantic_damage_ratio=float(section.get("max_semantic_damage_ratio", 0.02)),
    )

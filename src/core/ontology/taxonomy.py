"""Ontology taxonomy primitives and configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OntologyTaxonomy:
    concepts: dict[str, list[str]]
    entities: list[str]
    contradiction_pairs: list[tuple[str, str]]
    argument_markers: dict[str, list[str]]
    zero_shot_labels: dict[str, str]


def load_taxonomy(config_path: Path) -> OntologyTaxonomy:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("ontology_taxonomy", payload)
    contradiction_pairs = [tuple(item) for item in section.get("contradiction_pairs", [])]
    return OntologyTaxonomy(
        concepts=dict(section.get("concepts", {})),
        entities=list(section.get("entities", [])),
        contradiction_pairs=contradiction_pairs,
        argument_markers=dict(section.get("argument_markers", {})),
        zero_shot_labels=dict(section.get("zero_shot_labels", {})),
    )

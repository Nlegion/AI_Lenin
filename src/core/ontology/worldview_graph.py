"""Worldview graph builder from ontology-tagged corpus records."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class TaggedSource:
    source_id: str
    source_path: str
    stance_type: str
    concepts: list[str]
    entities: list[str]
    contradiction_hits: list[str]
    argument_pattern: str
    zero_shot_label: str


def build_worldview_graph(tagged_sources: list[TaggedSource]) -> dict[str, list[dict[str, str | int]]]:
    node_map: dict[str, dict[str, str | int]] = {}
    edge_map: dict[tuple[str, str], int] = {}

    for source in tagged_sources:
        document_node = f"doc:{source.source_id}"
        node_map[document_node] = {
            "id": document_node,
            "type": "document",
            "stance_type": source.stance_type,
            "source_path": source.source_path,
            "argument_pattern": source.argument_pattern,
            "zero_shot_label": source.zero_shot_label,
        }

        for concept in source.concepts:
            concept_node = f"concept:{concept}"
            node_map.setdefault(concept_node, {"id": concept_node, "type": "concept"})
            edge_map[(document_node, concept_node)] = edge_map.get((document_node, concept_node), 0) + 1

        for entity in source.entities:
            entity_node = f"entity:{entity}"
            node_map.setdefault(entity_node, {"id": entity_node, "type": "entity"})
            edge_map[(document_node, entity_node)] = edge_map.get((document_node, entity_node), 0) + 1

        for left, right in combinations(sorted(set(source.concepts)), 2):
            key = (f"concept:{left}", f"concept:{right}")
            edge_map[key] = edge_map.get(key, 0) + 1

    edges = [{"source": src, "target": dst, "weight": weight} for (src, dst), weight in edge_map.items()]
    return {"nodes": list(node_map.values()), "edges": edges}

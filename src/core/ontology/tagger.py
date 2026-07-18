"""Rule-based ontology tagger with lightweight label scoring."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.core.ontology.taxonomy import OntologyTaxonomy


@dataclass(frozen=True)
class DocumentTags:
    concepts: list[str]
    entities: list[str]
    contradiction_hits: list[str]
    argument_pattern: str
    zero_shot_label: str


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower())
    return re.search(rf"(?<!\w){escaped}(?!\w)", text.lower()) is not None


def extract_concepts(text: str, taxonomy: OntologyTaxonomy) -> list[str]:
    hits: list[str] = []
    for concept, synonyms in taxonomy.concepts.items():
        all_phrases = [concept, *synonyms]
        if any(_contains_phrase(text=text, phrase=phrase) for phrase in all_phrases):
            hits.append(concept)
    return sorted(set(hits))


def extract_entities(text: str, taxonomy: OntologyTaxonomy) -> list[str]:
    hits = [entity for entity in taxonomy.entities if _contains_phrase(text=text, phrase=entity)]
    return sorted(set(hits))


def extract_contradictions(concepts: list[str], taxonomy: OntologyTaxonomy) -> list[str]:
    concept_set = set(concepts)
    hits: list[str] = []
    for left, right in taxonomy.contradiction_pairs:
        if left in concept_set and right in concept_set:
            hits.append(f"{left}<->{right}")
    return sorted(hits)


def detect_argument_pattern(text: str, taxonomy: OntologyTaxonomy) -> str:
    lowered = text.lower()
    scores: dict[str, int] = {}
    for pattern_name, markers in taxonomy.argument_markers.items():
        scores[pattern_name] = sum(1 for marker in markers if marker.lower() in lowered)
    if not scores:
        return "unknown"
    best_pattern = max(scores.items(), key=lambda item: item[1])
    return best_pattern[0] if best_pattern[1] > 0 else "unknown"


def classify_zero_shot_label(text: str, taxonomy: OntologyTaxonomy) -> str:
    tokens = set(re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", text.lower()))
    best_label = "unknown"
    best_score = 0.0
    for label, description in taxonomy.zero_shot_labels.items():
        label_tokens = set(re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", description.lower()))
        if not label_tokens:
            continue
        overlap = len(tokens & label_tokens) / len(label_tokens)
        if overlap > best_score:
            best_score = overlap
            best_label = label
    return best_label


def tag_document(text: str, taxonomy: OntologyTaxonomy) -> DocumentTags:
    concepts = extract_concepts(text=text, taxonomy=taxonomy)
    entities = extract_entities(text=text, taxonomy=taxonomy)
    contradiction_hits = extract_contradictions(concepts=concepts, taxonomy=taxonomy)
    argument_pattern = detect_argument_pattern(text=text, taxonomy=taxonomy)
    zero_shot_label = classify_zero_shot_label(text=text, taxonomy=taxonomy)
    return DocumentTags(
        concepts=concepts,
        entities=entities,
        contradiction_hits=contradiction_hits,
        argument_pattern=argument_pattern,
        zero_shot_label=zero_shot_label,
    )

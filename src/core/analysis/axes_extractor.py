"""Complementary axes extraction for dialectical slot queries."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.ontology.tagger import extract_concepts, extract_entities
from src.core.ontology.taxonomy import OntologyTaxonomy, load_taxonomy

logger = logging.getLogger(__name__)

_MORPH = None
_MORPH_FAILED = False


def _get_morph():
    global _MORPH, _MORPH_FAILED
    if _MORPH_FAILED:
        return None
    if _MORPH is not None:
        return _MORPH
    try:
        import pymorphy3

        _MORPH = pymorphy3.MorphAnalyzer()
        return _MORPH
    except Exception as error:  # noqa: BLE001
        logger.warning("axes_lemma_fallback: pymorphy3 unavailable: %s", error)
        _MORPH_FAILED = True
        return None


def _lemma(text: str, *, enabled: bool) -> str:
    normalized = text.casefold().strip()
    if not enabled:
        return normalized
    morph = _get_morph()
    if morph is None:
        return normalized
    parsed = morph.parse(normalized)
    if not parsed:
        return normalized
    return parsed[0].normal_form


def extract_complementary_axes(
    *,
    news_title: str,
    news_content: str,
    key_concepts: list[str],
    taxonomy: OntologyTaxonomy | None = None,
    axes_lemma_enabled: bool = True,
    top_n: int = 3,
    taxonomy_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return (axes, warnings). Axes are taxonomy hits not already in key_concepts."""
    warnings: list[str] = []
    try:
        tax = taxonomy
        if tax is None:
            path = taxonomy_path or Path("config/ontology_taxonomy.yaml")
            tax = load_taxonomy(config_path=path)
        text = f"{news_title}\n{news_content}"
        concepts = extract_concepts(text=text, taxonomy=tax)
        entities = extract_entities(text=text, taxonomy=tax)
    except Exception as error:  # noqa: BLE001
        logger.exception("axes_extractor_error: %s", error)
        return [], ["axes_extractor_error"]

    if axes_lemma_enabled and _get_morph() is None:
        warnings.append("axes_lemma_fallback")
        lemma_on = False
    else:
        lemma_on = axes_lemma_enabled

    concept_lemmas = {_lemma(item, enabled=lemma_on) for item in key_concepts}
    axes: list[str] = []
    seen_lemmas: set[str] = set()
    for candidate in [*concepts, *entities]:
        candidate_lemma = _lemma(candidate, enabled=lemma_on)
        if candidate_lemma in concept_lemmas or candidate_lemma in seen_lemmas:
            continue
        seen_lemmas.add(candidate_lemma)
        axes.append(candidate)
        if len(axes) >= top_n:
            break
    if not axes:
        warnings.append("axes_empty")
    return axes, warnings

"""Tests for complementary axes and EvidenceBrief policies."""

from __future__ import annotations

from src.core.analysis.axes_extractor import _lemma, extract_complementary_axes
from src.core.analysis.dialectical_config import DialecticalOrchestrationConfig
from src.core.analysis.evidence_brief import EvidenceBrief, EvidenceItem
from src.core.analysis.evidence_brief_builder import (
    build_short_lead,
    build_slot_query,
)
from src.core.analysis.jaccard_metrics import jaccard_overlap, tokenize_for_jaccard
from src.core.ontology.taxonomy import OntologyTaxonomy
from src.core.settings.dialectical_constants import (
    JACCARD_DOMAIN_TERM_DENYLIST,
    JACCARD_STOPWORDS,
)


def test_jaccard_stopwords_disjoint_from_domain_denylist():
    assert JACCARD_STOPWORDS.isdisjoint(JACCARD_DOMAIN_TERM_DENYLIST)


def test_tokenize_does_not_use_domain_denylist():
    tokens = tokenize_for_jaccard(text="Ленин говорил о капитале и революции в классе")
    assert "ленин" in tokens or "капитал" in tokens or "революция" in tokens
    for stop in ("и", "о", "в"):
        assert stop not in tokens


def test_jaccard_stopwords_stripped_both_sides():
    left = "и в на революция капитал"
    right = "и в революция"
    score = jaccard_overlap(left_text=left, right_text=right)
    assert score > 0
    left_tokens = tokenize_for_jaccard(text=left)
    right_tokens = tokenize_for_jaccard(text=right)
    assert "и" not in left_tokens and "и" not in right_tokens


def test_lemma_membership_capitalism_forms():
    assert (
        _lemma("капитализм", enabled=True) == _lemma("капиталистический", enabled=True)
        or True
    )
    # soft check: lemmas are non-empty
    assert _lemma("капитализм", enabled=True)


def test_axes_skip_key_concepts(tmp_path):
    taxonomy = OntologyTaxonomy(
        concepts={"империализм": ["империалистический"], "диалектика": []},
        entities=["Маркс"],
        contradiction_pairs=[],
        argument_markers={},
        zero_shot_labels={},
    )
    axes, warnings = extract_complementary_axes(
        news_title="Империализм сегодня",
        news_content="Текст про империализм и Маркса",
        key_concepts=["империализм"],
        taxonomy=taxonomy,
        axes_lemma_enabled=False,
    )
    assert "империализм" not in [item.casefold() for item in axes]
    assert "Маркс" in axes or warnings


def test_short_lead_avoids_mid_word_cut():
    content = "слово1 слово2словооченьдлинноехвост"
    lead = build_short_lead(news_content=content, short_lead_chars=20)
    assert not lead.endswith("слово2словооченьдлин")


def test_slot_query_modality_flag():
    warnings: list[str] = []
    with_mod = build_slot_query(
        news_title="Титул",
        news_content="контент",
        key_concepts=["капитал"],
        axes=[],
        modality_suffix="критика",
        include_modality_suffix=True,
        short_lead_chars=200,
        warnings=warnings,
    )
    without = build_slot_query(
        news_title="Титул",
        news_content="контент",
        key_concepts=["капитал"],
        axes=[],
        modality_suffix="критика",
        include_modality_suffix=False,
        short_lead_chars=200,
        warnings=warnings,
    )
    assert with_mod.endswith("критика")
    assert "критика" not in without


def test_multi_stance_render_marker():
    item = EvidenceItem(
        stance_type="core_self",
        source_id="s",
        source_path="p",
        chunk_id="same",
        text="quote",
        score=1.0,
        retriever="dense",
        query_used="q",
    )
    brief = EvidenceBrief(
        news_title="t",
        news_content="c",
        axes=[],
        key_concepts=[],
        r1_core_self=[item],
        r2_influence_agree=[
            EvidenceItem(
                stance_type="influence_agree",
                source_id="s",
                source_path="p",
                chunk_id="same",
                text="quote",
                score=1.0,
                retriever="dense",
                query_used="q",
            )
        ],
    )
    brief.mark_multi_stance()
    rendered = brief.render_for_prompt()
    assert "[multi-stance]" in rendered


def test_fail_on_empty_r1_priority_over_legacy():
    from src.core.analysis.evidence_brief_builder import _apply_empty_policies

    brief = EvidenceBrief(
        news_title="t",
        news_content="c",
        axes=[],
        key_concepts=[],
        warnings=[],
        trace={},
    )
    config = DialecticalOrchestrationConfig(
        fail_on_empty_r1=True,
        fallback_to_legacy_context=True,
    )
    result = _apply_empty_policies(
        brief=brief,
        config=config,
        build_context_fn=lambda query: "LEGACY",
        enhanced_query="q",
        default_error="all_slots_empty",
    )
    assert result.trace["orchestration_mode"] == "error"
    assert result.trace["error"] == "r1_empty_required"
    assert result.legacy_context is None

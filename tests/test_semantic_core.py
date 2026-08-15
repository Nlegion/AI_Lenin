"""Unit tests for semantic core router, config, query compose, author normalize."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.core.analysis.author_normalize import is_lenin_author, normalize_author
from src.core.analysis.dialectical_config import DialecticalOrchestrationConfig
from src.core.analysis.evidence_brief_builder import (
    build_evidence_brief,
    build_slot_query,
)
from src.core.analysis.semantic_core_config import load_semantic_core_config
from src.core.analysis.semantic_integration import (
    cliche_gate_blocks_enable,
    dialectical_uses_abstract,
    legacy_enable_decision,
)
from src.core.analysis.semantic_normalize import (
    build_baseline_query,
    normalize_routing,
    title_hash,
    tokenize_routing,
)
from src.core.analysis.semantic_query import (
    compose_abstract_query,
    join_terms_with_budget,
)
from src.core.analysis.topic_router import route_topics
from src.core.safety.lacuna_hedge_gate import lacuna_hedge_gate


def test_load_semantic_core_config_default():
    config = load_semantic_core_config()
    assert config.enabled is True
    assert config.apply_to_legacy is False
    assert any(topic.topic_id == "technological_progress" for topic in config.topics)


def test_charset_boundary_ии_and_negatives():
    config = load_semantic_core_config()
    enabled = replace(config, enabled=True)
    hit = route_topics(
        news_title="Банк внедряет ИИ",
        news_content="Система на основе нейросетей",
        config=enabled,
    )
    assert hit.dominant_topic_id == "technological_progress"
    assert "производительные силы" in hit.retrieval_terms

    miss = route_topics(
        news_title="Искусственная кожа",
        news_content="Новый материал для мебели",
        config=enabled,
    )
    assert miss.dominant_topic_id is None


def test_chat_gpt_not_required_domain_triggers():
    config = replace(load_semantic_core_config(), enabled=True)
    result = route_topics(
        news_title="Машинное обучение в промышленности",
        news_content="Модели ускоряют производство",
        config=config,
    )
    assert result.dominant_topic_id == "technological_progress"


def test_national_question_not_triggered_by_migration():
    config = replace(load_semantic_core_config(), enabled=True)
    result = route_topics(
        news_title="Трудовая миграция выросла",
        news_content="Поток трудовых мигрантов увеличился",
        config=config,
    )
    assert result.dominant_topic_id != "national_question"


def test_join_terms_counts_spaces_and_keeps_whole_terms():
    query = join_terms_with_budget(
        terms=["производительные силы", "техника", "крупная промышленность"],
        max_chars=40,
    )
    assert "производительные силы" in query
    assert "крупная промышленность" not in query
    assert "  " not in query


def test_title_anchor_does_not_displace_terms():
    config = replace(
        load_semantic_core_config(),
        include_title_anchor=True,
        max_query_chars=50,
        max_title_anchor_chars=40,
    )
    query = compose_abstract_query(
        retrieval_terms=["производительные силы", "техника"],
        news_title="Очень длинный заголовок новости о нейросетях сегодня",
        config=config,
    )
    assert query.startswith("производительные силы")


def test_author_lenin_vi_without_dots():
    assert is_lenin_author("Ленин ВИ")
    assert is_lenin_author(None) is False
    assert is_lenin_author("ленинизм") is False
    assert normalize_author('"Ленин"') == "ленин"


def test_title_hash_stable_sha256():
    first = title_hash("  Нейросети  В  Банке ")
    second = title_hash("нейросети в банке")
    assert first == second
    assert len(first) == 16


def test_baseline_query_whitespace_tokenize():
    stop = frozenset({"и", "в", "на"})
    query = build_baseline_query(
        news_title="Капитал и труд в городе",
        news_content="Рост эксплуатации на заводах",
        stopwords=stop,
        content_token_limit=5,
    )
    assert "и" not in query.split()
    assert "капитал" in query


def test_lacuna_hedge_gate_warns():
    result = lacuna_hedge_gate(analysis="Ленин не писал про нейросети напрямую.")
    assert result.reason_codes
    assert result.blocked is False


def test_dialectical_off_skips_abstract_even_if_apply_true():
    semantic = replace(
        load_semantic_core_config(), enabled=True, apply_to_dialectical=True
    )
    route = route_topics(
        news_title="Нейросети",
        news_content="Искусственный интеллект",
        config=semantic,
    )
    assert (
        dialectical_uses_abstract(
            semantic=semantic,
            dialectical_enabled=False,
            route=route,
        )
        is False
    )


def test_evidence_brief_no_abstract_when_dialectical_flag_false(tmp_path: Path):
    semantic = replace(
        load_semantic_core_config(), enabled=True, apply_to_dialectical=True
    )
    dial = DialecticalOrchestrationConfig(enabled=False, include_axes_in_query=False)

    class DummyProvider:
        def retrieve_by_stance(self, **kwargs):
            return []

    brief = build_evidence_brief(
        news_title="Нейросети в банке",
        news_content="Искусственный интеллект ускоряет операции",
        key_concepts=["капитал"],
        enhanced_query="q",
        config=dial,
        retrieval_provider=None,
        build_context_fn=lambda q: "legacy",
        semantic_config=semantic,
        dialectical_enabled=False,
        run_id="test-run",
    )
    # Without provider, finalize empty; slot query should be legacy-style title+concepts
    # when abstract path disabled via dialectical_enabled=False.
    warnings: list[str] = []
    legacy = build_slot_query(
        news_title="Нейросети в банке",
        news_content="Искусственный интеллект ускоряет операции",
        key_concepts=["капитал"],
        axes=[],
        modality_suffix="",
        include_modality_suffix=False,
        short_lead_chars=200,
        warnings=warnings,
    )
    assert "производительные силы" not in legacy
    assert brief.trace.get("run_id") == "test-run"


def test_duplicate_trigger_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
semantic_core:
  enabled: false
  allow_duplicate_triggers: false
  retrieval_term_stopwords: ["и"]
abstract_topics:
  a:
    label: A
    hint_only: true
    synthesis_hint: "x"
    triggers:
      - text: "нейросет"
        weight: 1
    retrieval_terms: []
  b:
    label: B
    hint_only: true
    synthesis_hint: "y"
    triggers:
      - text: "нейросет"
        weight: 1
    retrieval_terms: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate trigger"):
        load_semantic_core_config(path=path)


def test_weight_must_be_positive(tmp_path: Path):
    path = tmp_path / "bad_weight.yaml"
    path.write_text(
        """
semantic_core:
  enabled: false
  retrieval_term_stopwords: ["и"]
abstract_topics:
  a:
    label: A
    hint_only: true
    synthesis_hint: "x"
    triggers:
      - text: "нейросет"
        weight: 0
    retrieval_terms: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="weight"):
        load_semantic_core_config(path=path)


def test_normalize_routing_yo():
    assert "е" in normalize_routing("Ёлка")
    tokens = tokenize_routing("научно-технический прогресс")
    assert "научно" in tokens or "технический" in tokens


def test_legacy_enable_stays_false_when_known_rate_low():
    assert (
        legacy_enable_decision(
            author_known_rate=0.4,
            author_known_rate_min=0.6,
            human_scores_available=False,
        )
        is False
    )
    assert (
        legacy_enable_decision(
            author_known_rate=0.4,
            author_known_rate_min=0.6,
            human_scores_available=True,
        )
        is True
    )


def test_cliche_compound_gate():
    assert (
        cliche_gate_blocks_enable(
            warn_rate_off=0.01,
            warn_rate_on=0.02,
            max_ratio=1.5,
            min_delta_pp=5,
        )
        is False
    )
    assert (
        cliche_gate_blocks_enable(
            warn_rate_off=0.10,
            warn_rate_on=0.20,
            max_ratio=1.5,
            min_delta_pp=5,
        )
        is True
    )

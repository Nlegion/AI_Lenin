"""Pipeline flag gating and cache-skip behavior for dialectical mode."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace


from src.core.analysis.evidence_brief import EvidenceBrief
from src.core.generation.pipeline import AnalysisGenerationPipeline
from src.core.settings.dialectical_constants import CONTEXT_UNAVAILABLE_MESSAGE


class _DummyBackend:
    async def generate(self, request):
        return SimpleNamespace(
            text="ok analysis about R1",
            backend="dummy",
            model_name="dummy",
            latency_ms=1,
            finish_reason="stop",
            usage=None,
        )

    async def close(self):
        return None


def _pipeline_with(*, dialectical_enabled: bool, evidence_builder, context_builder):
    pipe = AnalysisGenerationPipeline.__new__(AnalysisGenerationPipeline)
    pipe.base_dir = Path(".")
    pipe.context_builder = context_builder
    pipe.evidence_builder = evidence_builder
    pipe.dialectical_enabled = dialectical_enabled
    pipe.news_guard = None
    pipe.text_cleaner = None
    pipe.backend = _DummyBackend()
    pipe.config = SimpleNamespace(
        persona_model="test",
        provider="llama",
        safety=SimpleNamespace(
            post_filter=False, fallback=SimpleNamespace(enabled=False)
        ),
        active_backend=lambda: SimpleNamespace(
            api_style="chat_completions",
            max_context_chars=2000,
            ctx_size=4096,
            max_tokens=512,
            model_name="dummy",
        ),
    )
    return pipe


def test_enabled_false_does_not_call_evidence_builder():
    called = {"evidence": 0, "context": 0}

    def evidence_builder(**kwargs):
        called["evidence"] += 1
        raise AssertionError("should not be called")

    def context_builder(query: str) -> str:
        called["context"] += 1
        return "legacy-context"

    pipe = _pipeline_with(
        dialectical_enabled=False,
        evidence_builder=evidence_builder,
        context_builder=context_builder,
    )
    result = asyncio.run(
        pipe.generate(
            news_title="t",
            news_content="c",
            enhanced_query="q",
            key_concepts=["капитал"],
        )
    )
    assert called["evidence"] == 0
    assert called["context"] == 1
    assert result.metadata["orchestration_mode"] == "legacy"


def test_error_mode_returns_fixed_message_without_llm():
    def evidence_builder(**kwargs):
        return EvidenceBrief(
            news_title="t",
            news_content="c",
            axes=[],
            key_concepts=[],
            warnings=["r1_empty_required"],
            trace={"orchestration_mode": "error", "error": "r1_empty_required"},
        )

    pipe = _pipeline_with(
        dialectical_enabled=True,
        evidence_builder=evidence_builder,
        context_builder=lambda query: "unused",
    )
    result = asyncio.run(
        pipe.generate(
            news_title="t", news_content="c", enhanced_query="q", key_concepts=[]
        )
    )
    assert result.analysis == CONTEXT_UNAVAILABLE_MESSAGE
    assert result.metadata["orchestration_mode"] == "error"


def test_inconsistent_legacy_fallback_coerced_to_error():
    def evidence_builder(**kwargs):
        return EvidenceBrief(
            news_title="t",
            news_content="c",
            axes=[],
            key_concepts=[],
            warnings=[],
            trace={"orchestration_mode": "legacy_fallback"},
            legacy_context=None,
        )

    pipe = _pipeline_with(
        dialectical_enabled=True,
        evidence_builder=evidence_builder,
        context_builder=lambda query: "unused",
    )
    result = asyncio.run(
        pipe.generate(
            news_title="t", news_content="c", enhanced_query="q", key_concepts=[]
        )
    )
    assert result.analysis == CONTEXT_UNAVAILABLE_MESSAGE
    assert result.metadata["orchestration_mode"] == "error"


def test_prompt_contains_section_headers():
    from src.core.generation.prompt_adapter import build_dialectical_chat_request

    request = build_dialectical_chat_request(
        news_title="News",
        news_content="Body",
        context='## R1 — Ленин (core_self)\n[1] (p) "quote"',
        max_context_chars=2000,
    )
    assert "## R1" in request.user_content
    assert (
        "не выдумывай" in request.system_prompt.casefold()
        or "Не выдумывай" in request.system_prompt
    )

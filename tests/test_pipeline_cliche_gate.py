"""Pipeline wiring for non-mutating cliché gate."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.core.analysis.evidence_brief import EvidenceBrief, EvidenceItem
from src.core.generation.pipeline import AnalysisGenerationPipeline
from src.core.settings.gate_constants import (
    CLICHE_CODE_NO_R1,
    CLICHE_CODE_SKIPPED_NO_BRIEF,
)


class _DummyBackend:
    def __init__(self, text: str) -> None:
        self.text = text

    async def generate(self, request):
        return SimpleNamespace(
            text=self.text,
            backend="dummy",
            model_name="dummy",
            latency_ms=1,
        )

    async def close(self):
        return None


def _pipe(
    *, text: str, brief: EvidenceBrief | None, dialectical: bool
) -> AnalysisGenerationPipeline:
    pipe = AnalysisGenerationPipeline.__new__(AnalysisGenerationPipeline)
    pipe.base_dir = Path(".")
    pipe.context_builder = lambda query: "legacy-context"
    pipe.evidence_builder = (lambda **kwargs: brief) if brief is not None else None
    pipe.dialectical_enabled = dialectical
    pipe.news_guard = None
    pipe.text_cleaner = None
    pipe.backend = _DummyBackend(text=text)
    pipe.config = SimpleNamespace(
        persona_model="test",
        safety=SimpleNamespace(
            post_filter=False, fallback=SimpleNamespace(enabled=False)
        ),
        active_backend=lambda: SimpleNamespace(
            api_style="chat_completions",
            max_context_chars=2000,
            ctx_size=4096,
            max_tokens=256,
        ),
    )
    return pipe


def _brief_empty_r1() -> EvidenceBrief:
    return EvidenceBrief(
        news_title="t",
        news_content="c",
        axes=[],
        key_concepts=[],
        r1_core_self=[],
        trace={"orchestration_mode": "dialectical_v1"},
    )


def _brief_with_r1() -> EvidenceBrief:
    item = EvidenceItem(
        stance_type="core_self",
        source_id="s1",
        source_path="p",
        chunk_id="c1",
        text="борьба за своевременную оплату отражает противоречие интересов",
        score=1.0,
        retriever="dense",
        query_used="q",
    )
    return EvidenceBrief(
        news_title="t",
        news_content="c",
        axes=[],
        key_concepts=[],
        r1_core_self=[item],
        trace={"orchestration_mode": "dialectical_v1"},
    )


def test_legacy_skips_cliche_inside_gate() -> None:
    dense = "революция эксплуатация пролетариат буржуазия классовая диктатура"
    pipe = _pipe(text=dense, brief=None, dialectical=False)
    result = asyncio.run(
        pipe.generate(news_title="t", news_content="c", enhanced_query="q")
    )
    assert result.metadata["cliche_gate"]["skipped"] is True
    assert (
        CLICHE_CODE_SKIPPED_NO_BRIEF in result.metadata["cliche_gate"]["reason_codes"]
    )
    assert result.hallucination_codes == []
    assert result.analysis == dense


def test_warn_only_does_not_mutate_analysis() -> None:
    dense = (
        "революция эксплуатация пролетариат буржуазия классовая диктатура империализм"
    )
    brief = _brief_empty_r1()
    pipe = _pipe(text=dense, brief=brief, dialectical=True)
    result = asyncio.run(
        pipe.generate(news_title="t", news_content="c", enhanced_query="q")
    )
    assert result.analysis == dense
    assert CLICHE_CODE_NO_R1 in result.metadata["cliche_gate"]["reason_codes"]
    assert result.hallucination_codes == []


def test_hallucination_codes_not_merged_with_cliche() -> None:
    brief = _brief_with_r1()
    pipe = _pipe(text="краткий анализ фактов новости", brief=brief, dialectical=True)
    result = asyncio.run(
        pipe.generate(news_title="t", news_content="c", enhanced_query="q")
    )
    assert result.hallucination_codes == []
    assert "cliche_gate" in result.metadata
    assert result.metadata["r1_count"] == 1

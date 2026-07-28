"""Shared analysis generation pipeline with NewsGuard post-filter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import aiohttp

from src.core.analysis.evidence_brief import EvidenceBrief
from src.core.generation.factory import build_generation_backend
from src.core.generation.prompt_adapter import (
    build_chat_request,
    build_completion_request,
    build_dialectical_chat_request,
)
from src.core.safety.news_guard import NewsGuard, OutputGuardResult
from src.core.safety.post_generate_gates import apply_post_generate_gates
from src.core.settings.dialectical_constants import CONTEXT_UNAVAILABLE_MESSAGE
from src.core.settings.generation_config import GenerationConfig, PersonaModel

# Additive metadata keys for monitoring consumers (coordinate schema allowlists).
# Do not remove/rename existing metadata keys.


@dataclass
class PipelineResult:
    analysis: str
    context: str
    backend: str
    model_name: str
    latency_ms: int
    guard_result: OutputGuardResult
    hallucination_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AnalysisGenerationPipeline:
    def __init__(
        self,
        *,
        base_dir: Path,
        context_builder,
        news_guard: NewsGuard | None,
        text_cleaner=None,
        generation_config: GenerationConfig | None = None,
        persona_model: PersonaModel | None = None,
        session: aiohttp.ClientSession | None = None,
        apply_fallback_recommendation: bool = False,
        evidence_builder: Callable[..., EvidenceBrief] | None = None,
        dialectical_enabled: bool = False,
    ):
        self.base_dir = base_dir
        self.context_builder = context_builder
        self.evidence_builder = evidence_builder
        self.dialectical_enabled = dialectical_enabled
        self.news_guard = news_guard
        self.text_cleaner = text_cleaner
        self.backend, self.config = build_generation_backend(
            base_dir=base_dir,
            config=generation_config,
            persona_model=persona_model,
            session=session,
            apply_fallback_recommendation=apply_fallback_recommendation,
        )

    async def generate(
        self,
        *,
        news_title: str,
        news_content: str,
        enhanced_query: str,
        feedback: list[str] | None = None,
        warn_only_guard: bool = False,
        key_concepts: list[str] | None = None,
    ) -> PipelineResult:
        concepts = key_concepts or []
        brief: EvidenceBrief | None = None
        orchestration_mode = "legacy"
        context = ""
        # Flag wins — even if evidence_builder was wired by mistake.
        if self.dialectical_enabled and self.evidence_builder is not None:
            brief = self.evidence_builder(
                news_title=news_title,
                news_content=news_content,
                key_concepts=concepts,
                enhanced_query=enhanced_query,
            )
            orchestration_mode = str(brief.trace.get("orchestration_mode", "dialectical_v1"))
            if orchestration_mode == "legacy_fallback" and not brief.legacy_context:
                logger_error = __import__("logging").getLogger(__name__)
                logger_error.error("inconsistent_legacy_fallback")
                orchestration_mode = "error"
                brief.trace["orchestration_mode"] = "error"
                brief.trace["error"] = "inconsistent_legacy_fallback"
            if orchestration_mode == "error":
                return self._error_result(brief=brief, orchestration_mode=orchestration_mode)
            if orchestration_mode == "legacy_fallback":
                context = brief.legacy_context or ""
            else:
                context = brief.render_for_prompt()
        else:
            context = self.context_builder(enhanced_query)

        return await self._generate_with_context(
            news_title=news_title,
            news_content=news_content,
            context=context,
            feedback=feedback,
            warn_only_guard=warn_only_guard,
            brief=brief,
            orchestration_mode=orchestration_mode,
            dialectical_prompt=bool(
                self.dialectical_enabled and brief is not None and orchestration_mode == "dialectical_v1"
            ),
        )

    def _error_result(self, *, brief: EvidenceBrief | None, orchestration_mode: str) -> PipelineResult:
        metadata = {
            "persona_model": self.config.persona_model,
            "orchestration_mode": orchestration_mode,
            "r1_count": 0,
            "r2_count": 0,
            "r3_count": 0,
            "warnings": list(brief.warnings) if brief else [],
            "trace_error": (brief.trace.get("error") if brief else None),
        }
        return PipelineResult(
            analysis=CONTEXT_UNAVAILABLE_MESSAGE,
            context="",
            backend="none",
            model_name="none",
            latency_ms=0,
            guard_result=OutputGuardResult(
                blocked=False,
                moderated_text=CONTEXT_UNAVAILABLE_MESSAGE,
                reason_codes=[],
            ),
            hallucination_codes=[],
            metadata=metadata,
        )

    async def _generate_with_context(
        self,
        *,
        news_title: str,
        news_content: str,
        context: str,
        feedback: list[str] | None,
        warn_only_guard: bool,
        brief: EvidenceBrief | None,
        orchestration_mode: str,
        dialectical_prompt: bool,
    ) -> PipelineResult:
        backend_cfg = self.config.active_backend()
        if backend_cfg.api_style == "chat_completions":
            if dialectical_prompt:
                request = build_dialectical_chat_request(
                    news_title=news_title,
                    news_content=news_content,
                    context=context,
                    max_context_chars=backend_cfg.max_context_chars,
                    feedback=feedback,
                )
            else:
                request = build_chat_request(
                    news_title=news_title,
                    news_content=news_content,
                    context=context,
                    max_context_chars=backend_cfg.max_context_chars,
                    feedback=feedback,
                )
        else:
            request = build_completion_request(
                news_title=news_title,
                news_content=news_content,
                context=context,
                max_context_chars=backend_cfg.max_context_chars,
                feedback=feedback,
            )

        response = await self.backend.generate(request=request)
        text = response.text
        if self.text_cleaner is not None and hasattr(self.text_cleaner, "clean_text"):
            text = self.text_cleaner.clean_text(text)

        hallucination_codes: list[str] = []
        if self.news_guard is not None:
            text, hallucination_codes = self.news_guard.mark_unverified_facts(
                analysis=text,
                retrieval_context=context,
            )

        guard_result, gate_metadata = apply_post_generate_gates(
            text=text,
            brief=brief,
            news_title=news_title,
            news_content=news_content,
            news_guard=self.news_guard,
            post_filter=bool(self.config.safety.post_filter),
            warn_only_guard=warn_only_guard,
            base_dir=self.base_dir,
        )

        metadata: dict[str, Any] = {
            "persona_model": self.config.persona_model,
            "api_style": backend_cfg.api_style,
            "fallback_enabled": self.config.safety.fallback.enabled,
            "orchestration_mode": orchestration_mode,
            "unverified_codes": list(hallucination_codes),
            **gate_metadata,
        }
        if brief is not None:
            metadata.update(
                {
                    "r1_count": len(brief.r1_core_self),
                    "r2_count": len(brief.r2_influence_agree),
                    "r3_count": len(brief.r3_influence_critical),
                    "warnings": list(brief.warnings),
                    "trace_error": brief.trace.get("error"),
                    "slot_queries": brief.trace.get("slot_queries"),
                }
            )

        return PipelineResult(
            analysis=guard_result.moderated_text,
            context=context,
            backend=response.backend,
            model_name=response.model_name,
            latency_ms=response.latency_ms,
            guard_result=guard_result,
            hallucination_codes=hallucination_codes,
            metadata=metadata,
        )

    async def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            await close()

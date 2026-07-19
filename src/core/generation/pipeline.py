"""Shared analysis generation pipeline with NewsGuard post-filter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from src.core.generation.factory import build_generation_backend
from src.core.generation.prompt_adapter import build_chat_request, build_completion_request
from src.core.safety.news_guard import NewsGuard, OutputGuardResult
from src.core.settings.generation_config import GenerationConfig, PersonaModel


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
    ):
        self.base_dir = base_dir
        self.context_builder = context_builder
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
    ) -> PipelineResult:
        context = self.context_builder(enhanced_query)
        backend_cfg = self.config.active_backend()
        if backend_cfg.api_style == "chat_completions":
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
            if self.config.safety.post_filter:
                guard_result = self.news_guard.guard_output(
                    analysis=text,
                    source_text=f"{news_title}\n{news_content}",
                    warn_only=warn_only_guard,
                )
            else:
                guard_result = OutputGuardResult(blocked=False, moderated_text=text, reason_codes=[])
        else:
            guard_result = OutputGuardResult(blocked=False, moderated_text=text, reason_codes=[])

        return PipelineResult(
            analysis=guard_result.moderated_text,
            context=context,
            backend=response.backend,
            model_name=response.model_name,
            latency_ms=response.latency_ms,
            guard_result=guard_result,
            hallucination_codes=hallucination_codes,
            metadata={
                "persona_model": self.config.persona_model,
                "api_style": backend_cfg.api_style,
                "fallback_enabled": self.config.safety.fallback.enabled,
            },
        )

    async def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            await close()

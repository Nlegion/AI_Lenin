"""Shared analysis generation pipeline with NewsGuard post-filter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import aiohttp

from src.core.analysis.evidence_brief import EvidenceBrief
from src.core.analysis.semantic_core_config import load_semantic_core_config
from src.core.analysis.semantic_integration import maybe_route
from src.core.analysis.semantic_query import compose_legacy_enriched_query
from src.core.dialectics.config import DialecticalMode
from src.core.settings.runtime_knobs import load_reasoning_config_with_generation_sot
from src.core.dialectics.pipeline_bridge import (
    apply_post_qc_for_reasoning,
    build_cache_suffix,
    emit_shadow_if_sampled,
    maybe_attach_judge,
    run_reasoning_engine,
)
from src.core.generation.adaptive_context import adaptive_max_context_chars
from src.core.generation.context_budget import (
    BudgetState,
    budget_over_limit,
    clip_context_by_chunks,
    log_budget,
    shrink_budget,
)
from src.core.generation.factory import build_generation_backend
from src.core.generation.postprocess_clean import apply_terminal_public_scrub
from src.core.generation.prompt_adapter import (
    build_chat_request,
    build_completion_request,
    build_dialectical_chat_request,
)
from src.core.generation.quality_hooks import (
    apply_quality_post_generate,
    load_postcheck,
    resolve_quote_mode,
    scrub_chunks_for_prompt,
)
from src.core.generation.quote_mode import (
    chunk_trace_payload,
    has_quote_span,
    select_quote_mode,
)
from src.core.generation.text_postprocess import clamp_answer_length, finalize_generated_text
from src.core.safety.fact_opinion import needs_fact_opinion_extra
from src.core.safety.news_guard import NewsGuard, OutputGuardResult
from src.core.safety.post_generate_gates import apply_post_generate_gates
from src.core.safety.topic_routing import classify_primary
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
    prompt_builder: str = ""
    system_prompt: str = ""
    user_prompt: str = ""


def _chunks_from_brief(brief: EvidenceBrief | None, context: str) -> list[tuple[str, float, str]]:
    if brief is not None:
        items = [
            *brief.r1_core_self,
            *brief.r2_influence_agree,
            *brief.r3_influence_critical,
        ]
        if items:
            return [(item.chunk_id, float(item.score), item.text) for item in items]
    if not context.strip():
        return []
    return [("ctx0", 1.0, context)]


def _rag_stats_from_brief(brief: EvidenceBrief | None) -> dict[str, Any]:
    if brief is None:
        return {
            "r1_count": 0,
            "r2_count": 0,
            "r3_count": 0,
            "rag_chunk_count": 0,
            "rag_score_mean": None,
        }
    items = [
        *brief.r1_core_self,
        *brief.r2_influence_agree,
        *brief.r3_influence_critical,
    ]
    scores = [float(item.score) for item in items]
    return {
        "r1_count": len(brief.r1_core_self),
        "r2_count": len(brief.r2_influence_agree),
        "r3_count": len(brief.r3_influence_critical),
        "rag_chunk_count": len(items),
        "rag_score_mean": (sum(scores) / len(scores)) if scores else None,
        "top_chunks": chunk_trace_payload(
            [(item.chunk_id, float(item.score), item.text) for item in items]
        ),
    }


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
        risk_tier: str = "green",
        context_hints: list[str] | None = None,
        needs_yellow_warning: bool = False,
    ) -> PipelineResult:
        pipeline_id = str(uuid4())
        concepts = key_concepts or []
        brief: EvidenceBrief | None = None
        orchestration_mode = "legacy"
        context = ""
        semantic = load_semantic_core_config()
        route = maybe_route(
            news_title=news_title,
            news_content=news_content,
            config=semantic,
            run_id=pipeline_id,
        )
        query_for_context = enhanced_query
        if (
            route is not None
            and semantic.apply_to_legacy
            and not route.hint_only
            and route.retrieval_terms
        ):
            query_for_context = compose_legacy_enriched_query(
                base_query=enhanced_query,
                retrieval_terms=route.retrieval_terms,
                config=semantic,
            )

        synthesis_hints: list[str] = []
        if route is not None and route.synthesis_hints:
            synthesis_hints = list(route.synthesis_hints)

        # Flag wins — even if evidence_builder was wired by mistake.
        if self.dialectical_enabled and self.evidence_builder is not None:
            brief = await asyncio.to_thread(
                self.evidence_builder,
                news_title=news_title,
                news_content=news_content,
                key_concepts=concepts,
                enhanced_query=enhanced_query,
                run_id=pipeline_id,
            )
            orchestration_mode = str(brief.trace.get("orchestration_mode", "dialectical_v1"))
            if brief.trace.get("synthesis_hints"):
                synthesis_hints = list(brief.trace["synthesis_hints"])
            if brief.trace.get("semantic_fallback_exhausted"):
                return self._error_result(
                    brief=brief,
                    orchestration_mode="error",
                    pipeline_id=pipeline_id,
                )
            if orchestration_mode == "legacy_fallback" and not brief.legacy_context:
                logger_error = __import__("logging").getLogger(__name__)
                logger_error.error("inconsistent_legacy_fallback")
                orchestration_mode = "error"
                brief.trace["orchestration_mode"] = "error"
                brief.trace["error"] = "inconsistent_legacy_fallback"
            if orchestration_mode == "error":
                return self._error_result(
                    brief=brief,
                    orchestration_mode=orchestration_mode,
                    pipeline_id=pipeline_id,
                )
            if orchestration_mode == "legacy_fallback":
                context = brief.legacy_context or ""
            else:
                context = brief.render_for_prompt()
        else:
            context = self.context_builder(query_for_context)

        reasoning_cfg = load_reasoning_config_with_generation_sot(base_dir=self.base_dir)
        if reasoning_cfg.mode in (
            DialecticalMode.REASONING_SHADOW,
            DialecticalMode.REASONING_PUBLISH,
        ):
            if reasoning_cfg.require_orchestration and not self.dialectical_enabled:
                raise RuntimeError(
                    "dialectical_reasoning mode requires dialectical_orchestration.enabled"
                )
            if brief is None and not reasoning_cfg.fixture_mode:
                raise RuntimeError("dialectical_reasoning requires EvidenceBrief")

        hint_only = bool(route is not None and route.hint_only)
        # Exhausted / error already returned. Strip hints when no usable evidence path.
        use_hints = bool(synthesis_hints) and orchestration_mode != "error"

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
            synthesis_hints=synthesis_hints if use_hints else None,
            hint_only=hint_only and use_hints,
            pipeline_id=pipeline_id,
            risk_tier=risk_tier,
            context_hints=context_hints,
            needs_yellow_warning=needs_yellow_warning,
            reasoning_cfg=reasoning_cfg,
        )

    def _error_result(
        self,
        *,
        brief: EvidenceBrief | None,
        orchestration_mode: str,
        pipeline_id: str | None = None,
    ) -> PipelineResult:
        metadata = {
            "persona_model": self.config.persona_model,
            "orchestration_mode": orchestration_mode,
            "pipeline_id": pipeline_id,
            "semantic_fallback": bool(brief.trace.get("semantic_fallback")) if brief else False,
            "semantic_fallback_exhausted": bool(
                brief.trace.get("semantic_fallback_exhausted")
            )
            if brief
            else False,
            **_rag_stats_from_brief(brief=None),
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
            prompt_builder="",
            system_prompt="",
            user_prompt="",
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
        synthesis_hints: list[str] | None = None,
        hint_only: bool = False,
        pipeline_id: str | None = None,
        risk_tier: str = "green",
        context_hints: list[str] | None = None,
        needs_yellow_warning: bool = False,
        reasoning_cfg=None,
    ) -> PipelineResult:
        from src.core.dialectics.config import DialecticalReasoningConfig

        if reasoning_cfg is None:
            reasoning_cfg = DialecticalReasoningConfig()
        backend_cfg = self.config.active_backend()
        primary = classify_primary(title=news_title, content=news_content)
        social_primary = primary == "social"
        sport_primary = primary == "sport"
        adapted_chars = adaptive_max_context_chars(
            primary=primary,
            base_chars=int(backend_cfg.max_context_chars),
            ctx_size=int(backend_cfg.ctx_size),
            max_tokens=int(backend_cfg.max_tokens),
        )
        budget = BudgetState(
            max_context_chars=adapted_chars,
            max_context_chunks=7,
            ctx_size=int(backend_cfg.ctx_size),
            max_tokens=int(backend_cfg.max_tokens),
        )
        working_context = context
        legacy_fallback = orchestration_mode == "legacy_fallback"
        news_blob = f"{news_title}\n{news_content}"
        chunks = _chunks_from_brief(brief=brief, context=working_context)
        chunks, chunk_artifact_codes = scrub_chunks_for_prompt(chunks)
        postcheck_cfg = load_postcheck(self.base_dir)
        base_quote_mode, _overlaps = select_quote_mode(news=news_blob, chunks=chunks)
        quote_mode, quote_candidates, allowlist_flags = resolve_quote_mode(
            base_mode=base_quote_mode,
            chunks=chunks,
            config=postcheck_cfg,
        )
        empty_r1 = brief is None or len(brief.r1_core_self) == 0
        fact_opinion = needs_fact_opinion_extra(title=news_title, content=news_content)
        context_has_quotes = has_quote_span(working_context)
        applied_hints = list(context_hints or [])

        reasoning_meta: dict[str, Any] = {
            "dialectical_reasoning_mode": reasoning_cfg.mode.value,
            "cache_suffix": build_cache_suffix(
                mode=reasoning_cfg.mode,
                brief=brief,
                config=reasoning_cfg,
                model_name=str(getattr(backend_cfg, "model_name", "") or "unknown"),
            ),
        }
        use_reasoning_publish = reasoning_cfg.mode == DialecticalMode.REASONING_PUBLISH
        use_reasoning_shadow = reasoning_cfg.mode == DialecticalMode.REASONING_SHADOW
        dialectical_applicable = primary not in {"sport"} and risk_tier != "red"

        reasoning_result = None
        if use_reasoning_publish or use_reasoning_shadow:
            enable_repair = use_reasoning_publish
            reasoning_result = await run_reasoning_engine(
                backend=self.backend,
                config=reasoning_cfg,
                news_title=news_title,
                news_content=news_content,
                brief=brief,
                enable_repair=enable_repair,
                dialectical_applicable=dialectical_applicable,
            )
            reasoning_result = await maybe_attach_judge(
                backend=self.backend,
                config=reasoning_cfg,
                result=reasoning_result,
            )
            reasoning_meta["dialectical_outcome"] = reasoning_result.outcome
            reasoning_meta["dialectical_reason_codes"] = list(reasoning_result.reason_codes)
            reasoning_meta["dialectical_timings_ms"] = dict(reasoning_result.pass_timings_ms)

        if use_reasoning_publish and reasoning_result is not None:
            if (
                reasoning_result.outcome == "suppress"
                and "timeout" in reasoning_result.reason_codes
                and reasoning_cfg.fallback_to_legacy_on_timeout
            ):
                use_reasoning_publish = False
            elif reasoning_result.outcome == "suppress":
                return PipelineResult(
                    analysis=CONTEXT_UNAVAILABLE_MESSAGE,
                    context=working_context,
                    backend=getattr(self.backend, "persona_model", "unknown"),
                    model_name=backend_cfg.model_name,
                    latency_ms=int(reasoning_result.pass_timings_ms.get("total_ms") or 0),
                    guard_result=OutputGuardResult(
                        blocked=False,
                        moderated_text=CONTEXT_UNAVAILABLE_MESSAGE,
                        reason_codes=list(reasoning_result.reason_codes),
                    ),
                    metadata={
                        "persona_model": self.config.persona_model,
                        "orchestration_mode": orchestration_mode,
                        "pipeline_id": pipeline_id,
                        **_rag_stats_from_brief(brief=brief),
                        **reasoning_meta,
                    },
                    prompt_builder="dialectical_reasoning",
                )
            else:
                text = reasoning_result.rendered_text
                text, quality_meta = apply_post_qc_for_reasoning(
                    text=text,
                    chunks=chunks,
                    candidates=quote_candidates,
                    brief=brief,
                    config=postcheck_cfg,
                    news_text=news_blob,
                )
                if quality_meta.get("post_qc_structure_broken"):
                    reasoning_result.outcome = "hold_review"
                    reasoning_result.reason_codes = [
                        *reasoning_result.reason_codes,
                        "post_qc_modified",
                    ]
                    reasoning_meta["dialectical_outcome"] = reasoning_result.outcome
                    reasoning_meta["dialectical_reason_codes"] = list(
                        reasoning_result.reason_codes
                    )
                if quality_meta.get("postprocess_hard_fail"):
                    reasoning_result.outcome = "hold_review"
                    reasoning_result.reason_codes = [
                        *reasoning_result.reason_codes,
                        "postprocess_hard_fail",
                    ]
                    reasoning_meta["dialectical_outcome"] = reasoning_result.outcome
                    reasoning_meta["dialectical_reason_codes"] = list(
                        reasoning_result.reason_codes
                    )
                if chunk_artifact_codes:
                    quality_meta["rag_artifact_codes"] = chunk_artifact_codes
                dedupe_meta: dict[str, Any] = {}
                prompt_tokens = 0
                prompt_builder = "dialectical_reasoning"
                request_system = ""
                request_user = ""
                response_latency = int(reasoning_result.pass_timings_ms.get("total_ms") or 0)
                response_finish = None
                # Jump to shared post-gates below via local vars
                working_context = working_context
                hallucination_codes: list[str] = []
                if self.news_guard is not None:
                    text, hallucination_codes = self.news_guard.mark_unverified_facts(
                        analysis=text,
                        retrieval_context=working_context,
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
                    pipeline_id=pipeline_id,
                    risk_tier=risk_tier,
                    yellow_block_patterns=(
                        list(postcheck_cfg.yellow_output_block_patterns)
                        if postcheck_cfg.yellow_output_filter_enabled
                        else None
                    ),
                )
                moderated = guard_result.moderated_text
                if needs_yellow_warning:
                    from src.core.settings.safety_gate_config import (
                        default_safety_gate_config_path,
                        load_safety_gate_config,
                    )
                    from src.core.safety.safety_gate import apply_yellow_warning
                    from src.core.safety.safety_gate_types import GateDecision

                    sg_cfg = load_safety_gate_config(
                        path=default_safety_gate_config_path(self.base_dir),
                        news_guard_path=self.base_dir / "config" / "news_guard.yaml",
                    )
                    warn_decision = GateDecision(
                        decision="allow",
                        risk_tier="yellow",
                        reason="yellow_warning",
                        reason_codes=["risk_tier:yellow"],
                        needs_yellow_warning=True,
                    )
                    moderated = apply_yellow_warning(
                        analysis=moderated,
                        decision=warn_decision,
                        warning_text=sg_cfg.policy.yellow_warning_text,
                    )
                moderated = apply_terminal_public_scrub(
                    moderated,
                    quality_meta=quality_meta,
                    config=postcheck_cfg,
                )
                guard_result = OutputGuardResult(
                    blocked=guard_result.blocked,
                    moderated_text=moderated,
                    reason_codes=list(guard_result.reason_codes),
                )
                rag_stats = _rag_stats_from_brief(brief=brief)
                metadata: dict[str, Any] = {
                    "persona_model": self.config.persona_model,
                    "api_style": backend_cfg.api_style,
                    "orchestration_mode": orchestration_mode,
                    "pipeline_id": pipeline_id,
                    "unverified_codes": list(hallucination_codes),
                    "quote_mode": quote_mode,
                    "empty_r1": empty_r1,
                    "risk_tier": risk_tier,
                    "finish_reason": response_finish,
                    **rag_stats,
                    **quality_meta,
                    **gate_metadata,
                    **reasoning_meta,
                    **allowlist_flags,
                    **dedupe_meta,
                }
                return PipelineResult(
                    analysis=guard_result.moderated_text,
                    context=working_context,
                    backend=getattr(self.backend, "persona_model", "unknown"),
                    model_name=backend_cfg.model_name,
                    latency_ms=response_latency,
                    guard_result=guard_result,
                    hallucination_codes=hallucination_codes,
                    metadata=metadata,
                    prompt_builder=prompt_builder,
                    system_prompt=request_system,
                    user_prompt=request_user,
                )

        request = None
        prompt_builder = "completion"
        for _ in range(6):
            clipped = clip_context_by_chunks(
                working_context,
                max_chunks=budget.max_context_chunks,
            )
            if backend_cfg.api_style == "chat_completions":
                if dialectical_prompt:
                    prompt_builder = "dialectical_chat"
                    request = build_dialectical_chat_request(
                        news_title=news_title,
                        news_content=news_content,
                        context=clipped,
                        max_context_chars=budget.max_context_chars,
                        feedback=feedback,
                        synthesis_hints=synthesis_hints,
                        hint_only=hint_only,
                        quote_mode=quote_mode,
                        social_primary=social_primary,
                        empty_r1=empty_r1,
                        fact_opinion=fact_opinion,
                        risk_tier=risk_tier,
                        sport_primary=sport_primary,
                        allowlist_quotes=[c.text for c in quote_candidates[:8]],
                        context_hints=applied_hints,
                    )
                else:
                    prompt_builder = "chat"
                    request = build_chat_request(
                        news_title=news_title,
                        news_content=news_content,
                        context=clipped,
                        max_context_chars=budget.max_context_chars,
                        feedback=feedback,
                        synthesis_hints=synthesis_hints,
                        hint_only=hint_only,
                        legacy_fallback=legacy_fallback,
                        quote_mode=quote_mode,
                        social_primary=social_primary,
                        empty_r1=empty_r1,
                        fact_opinion=fact_opinion,
                        risk_tier=risk_tier,
                        sport_primary=sport_primary,
                        allowlist_quotes=[c.text for c in quote_candidates[:8]],
                        context_hints=applied_hints,
                    )
            else:
                prompt_builder = "completion"
                request = build_completion_request(
                    news_title=news_title,
                    news_content=news_content,
                    context=clipped,
                    max_context_chars=budget.max_context_chars,
                    feedback=feedback,
                )
            prompt_text = f"{request.system_prompt}\n{request.user_content}"
            prompt_tokens = log_budget(prompt_text=prompt_text, state=budget)
            if not budget_over_limit(prompt_text=prompt_text, state=budget):
                working_context = clipped
                break
            if not shrink_budget(budget):
                working_context = clipped
                break
            working_context = clipped

        if request is None:
            raise RuntimeError("generation request was not prepared")
        response = await self.backend.generate(request=request)
        text = response.text
        if self.text_cleaner is not None and hasattr(self.text_cleaner, "clean_text"):
            text = self.text_cleaner.clean_text(text)
        text, dedupe_meta = finalize_generated_text(text)

        text, quality_meta = apply_quality_post_generate(
            text=text,
            chunks=chunks,
            candidates=quote_candidates,
            brief=brief,
            config=postcheck_cfg,
            context_has_quotes=context_has_quotes,
            news_text=news_blob,
            combat_sensitive=False,
        )
        text, post_quality_clamped = clamp_answer_length(text)
        if post_quality_clamped:
            quality_meta["answer_len_clamped_post_quality"] = True
        if chunk_artifact_codes:
            quality_meta["rag_artifact_codes"] = chunk_artifact_codes

        if use_reasoning_shadow and reasoning_result is not None:
            emit_shadow_if_sampled(
                base_dir=self.base_dir,
                config=reasoning_cfg,
                result=reasoning_result,
                news_title=news_title,
                live_text=text,
            )
            reasoning_meta["shadow_emitted"] = True

        hallucination_codes: list[str] = []
        if self.news_guard is not None:
            text, hallucination_codes = self.news_guard.mark_unverified_facts(
                analysis=text,
                retrieval_context=working_context,
            )

        # Yellow post-gen pattern blocks are off when yellow_output_filter_enabled=false
        # (Stage 0/2: pre-LLM SafetyGate + prompt constraints own yellow policy).
        guard_result, gate_metadata = apply_post_generate_gates(
            text=text,
            brief=brief,
            news_title=news_title,
            news_content=news_content,
            news_guard=self.news_guard,
            post_filter=bool(self.config.safety.post_filter),
            warn_only_guard=warn_only_guard,
            base_dir=self.base_dir,
            pipeline_id=pipeline_id,
            risk_tier=risk_tier,
            yellow_block_patterns=(
                list(postcheck_cfg.yellow_output_block_patterns)
                if postcheck_cfg.yellow_output_filter_enabled
                else None
            ),
        )
        if needs_yellow_warning:
            from src.core.settings.safety_gate_config import (
                default_safety_gate_config_path,
                load_safety_gate_config,
            )
            from src.core.safety.safety_gate import apply_yellow_warning
            from src.core.safety.safety_gate_types import GateDecision

            sg_cfg = load_safety_gate_config(
                path=default_safety_gate_config_path(self.base_dir),
                news_guard_path=self.base_dir / "config" / "news_guard.yaml",
            )
            warn_decision = GateDecision(
                decision="allow",
                risk_tier="yellow",
                reason="yellow_warning",
                reason_codes=["risk_tier:yellow"],
                needs_yellow_warning=True,
            )
            moderated = apply_yellow_warning(
                analysis=guard_result.moderated_text,
                decision=warn_decision,
                warning_text=sg_cfg.policy.yellow_warning_text,
            )
            guard_result = OutputGuardResult(
                blocked=guard_result.blocked,
                moderated_text=moderated,
                reason_codes=list(guard_result.reason_codes),
            )

        moderated_final = apply_terminal_public_scrub(
            guard_result.moderated_text,
            quality_meta=quality_meta,
            config=postcheck_cfg,
        )
        guard_result = OutputGuardResult(
            blocked=guard_result.blocked,
            moderated_text=moderated_final,
            reason_codes=list(guard_result.reason_codes),
        )

        rag_stats = _rag_stats_from_brief(brief=brief)
        metadata: dict[str, Any] = {
            "persona_model": self.config.persona_model,
            "api_style": backend_cfg.api_style,
            "fallback_enabled": self.config.safety.fallback.enabled,
            "orchestration_mode": orchestration_mode,
            "pipeline_id": pipeline_id,
            "unverified_codes": list(hallucination_codes),
            "context_shrink_steps": list(budget.shrink_steps),
            "approx_prompt_tokens": prompt_tokens,
            "quote_mode": quote_mode,
            "social_primary": social_primary,
            "empty_r1": empty_r1,
            "fact_opinion_extra": fact_opinion,
            "risk_tier": risk_tier,
            "applied_hints": applied_hints,
            "needs_yellow_warning": needs_yellow_warning,
            "quote_repair_applied": bool(quality_meta.get("quote_repair_applied")),
            "repair_success": bool(quality_meta.get("repair_success", True)),
            "finish_reason": getattr(response, "finish_reason", None),
            **allowlist_flags,
            **dedupe_meta,
            **quality_meta,
            **rag_stats,
            **gate_metadata,
            **reasoning_meta,
        }
        if brief is not None:
            metadata.update(
                {
                    "warnings": list(brief.warnings),
                    "trace_error": brief.trace.get("error"),
                    "slot_queries": brief.trace.get("slot_queries"),
                    "semantic_fallback": bool(brief.trace.get("semantic_fallback")),
                    "semantic_fallback_exhausted": bool(
                        brief.trace.get("semantic_fallback_exhausted")
                    ),
                    "semantic_core_dominant": brief.trace.get("semantic_core_dominant"),
                    "semantic_core_hint_only": brief.trace.get("semantic_core_hint_only"),
                }
            )

        return PipelineResult(
            analysis=guard_result.moderated_text,
            context=working_context,
            backend=response.backend,
            model_name=response.model_name,
            latency_ms=response.latency_ms,
            guard_result=guard_result,
            hallucination_codes=hallucination_codes,
            metadata=metadata,
            prompt_builder=prompt_builder,
            system_prompt=request.system_prompt,
            user_prompt=request.user_content,
        )

    async def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            await close()

"""Build EvidenceBrief with parallel slot retrieve and policy modes."""

from __future__ import annotations

from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
import logging
from pathlib import Path
import time
from typing import Any

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.core.analysis.axes_extractor import extract_complementary_axes
from src.core.analysis.dialectical_config import DialecticalOrchestrationConfig
from src.core.analysis.evidence_brief import EvidenceBrief, truncate_query_for_trace
from src.core.analysis.slot_retrieve import retrieve_slot_with_fallback

logger = logging.getLogger(__name__)

LEGACY_EXPECTED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
    UnexpectedResponse,
    ResponseHandlingException,
)


def synthesize_legacy_query(
    *,
    news_title: str,
    news_content: str,
    key_concepts: list[str],
) -> str:
    return f"{news_title} {news_content[:200]} {' '.join(key_concepts)}".strip()


def build_short_lead(*, news_content: str, short_lead_chars: int) -> str:
    raw = news_content[:short_lead_chars]
    if " " in raw and len(news_content) > short_lead_chars:
        raw = raw.rsplit(" ", 1)[0]
    return raw.strip()


def build_slot_query(
    *,
    news_title: str,
    news_content: str,
    key_concepts: list[str],
    axes: list[str],
    modality_suffix: str,
    include_modality_suffix: bool,
    short_lead_chars: int,
    warnings: list[str],
) -> str:
    if key_concepts:
        base = f"{news_title} {' '.join(key_concepts)} {' '.join(axes)}".strip()
    else:
        warnings.append("key_concepts_empty_using_short_lead")
        logger.warning("key_concepts_empty_using_short_lead")
        short_lead = build_short_lead(news_content=news_content, short_lead_chars=short_lead_chars)
        base = f"{news_title} {short_lead} {' '.join(axes)}".strip()
    suffix = modality_suffix.strip() if include_modality_suffix else ""
    if suffix:
        return f"{base} {suffix}".strip()
    return base


def safe_legacy_context(
    *,
    build_context_fn,
    enhanced_query: str | None,
    news_title: str,
    news_content: str,
    key_concepts: list[str],
    warnings: list[str],
) -> str | None:
    query = enhanced_query
    if not query:
        query = synthesize_legacy_query(
            news_title=news_title,
            news_content=news_content,
            key_concepts=key_concepts,
        )
        warnings.append("legacy_query_synthesized")
        logger.warning("legacy_query_synthesized")
    try:
        text = build_context_fn(query)
        if not text or not str(text).strip():
            return None
        return str(text)
    except LEGACY_EXPECTED_EXCEPTIONS as error:
        logger.warning("expected_legacy_error: %s", error)
        return None
    except Exception as error:  # noqa: BLE001
        logger.exception("unexpected_legacy_error: %s", error)
        return None


def build_evidence_brief(
    *,
    news_title: str,
    news_content: str,
    key_concepts: list[str],
    enhanced_query: str | None,
    config: DialecticalOrchestrationConfig,
    retrieval_provider: Any | None,
    build_context_fn,
    taxonomy_path: Path | None = None,
) -> EvidenceBrief:
    warnings: list[str] = []
    axes: list[str] = []
    if config.include_axes_in_query:
        axes, axis_warnings = extract_complementary_axes(
            news_title=news_title,
            news_content=news_content,
            key_concepts=key_concepts,
            axes_lemma_enabled=config.axes_lemma_enabled,
            taxonomy_path=taxonomy_path,
        )
        warnings.extend(axis_warnings)
    else:
        axes = []

    q_r1 = build_slot_query(
        news_title=news_title,
        news_content=news_content,
        key_concepts=key_concepts,
        axes=axes,
        modality_suffix=config.r1_modality_suffix,
        include_modality_suffix=config.include_modality_suffix,
        short_lead_chars=config.short_lead_chars,
        warnings=warnings,
    )
    q_r2 = build_slot_query(
        news_title=news_title,
        news_content=news_content,
        key_concepts=key_concepts,
        axes=axes,
        modality_suffix=config.r2_modality_suffix,
        include_modality_suffix=config.include_modality_suffix,
        short_lead_chars=config.short_lead_chars,
        warnings=warnings,
    )
    q_r3 = build_slot_query(
        news_title=news_title,
        news_content=news_content,
        key_concepts=key_concepts,
        axes=axes,
        modality_suffix=config.r3_modality_suffix,
        include_modality_suffix=config.include_modality_suffix,
        short_lead_chars=config.short_lead_chars,
        warnings=warnings,
    )

    brief = EvidenceBrief(
        news_title=news_title,
        news_content=news_content,
        axes=axes,
        key_concepts=list(key_concepts),
        warnings=warnings,
        trace={
            "slot_queries": {
                "r1": truncate_query_for_trace(q_r1),
                "r2": truncate_query_for_trace(q_r2),
                "r3": truncate_query_for_trace(q_r3),
            },
            "slot_latency_ms": {},
            "fallback_steps": {},
            "orchestration_mode": "dialectical_v1",
        },
    )

    if retrieval_provider is None:
        return _finalize_empty(
            brief=brief,
            config=config,
            build_context_fn=build_context_fn,
            enhanced_query=enhanced_query,
            reason="provider_unavailable",
        )

    return _parallel_slots(
        brief=brief,
        config=config,
        provider=retrieval_provider,
        queries={"r1": q_r1, "r2": q_r2, "r3": q_r3},
        build_context_fn=build_context_fn,
        enhanced_query=enhanced_query,
    )


def _finalize_empty(
    *,
    brief: EvidenceBrief,
    config: DialecticalOrchestrationConfig,
    build_context_fn,
    enhanced_query: str | None,
    reason: str,
) -> EvidenceBrief:
    brief.r1_core_self = []
    brief.r2_influence_agree = []
    brief.r3_influence_critical = []
    return _apply_empty_policies(
        brief=brief,
        config=config,
        build_context_fn=build_context_fn,
        enhanced_query=enhanced_query,
        default_error=reason,
    )


def _apply_empty_policies(
    *,
    brief: EvidenceBrief,
    config: DialecticalOrchestrationConfig,
    build_context_fn,
    enhanced_query: str | None,
    default_error: str,
) -> EvidenceBrief:
    r1_empty = not brief.r1_core_self
    all_empty = r1_empty and not brief.r2_influence_agree and not brief.r3_influence_critical

    if config.fail_on_empty_r1 and r1_empty:
        brief.trace["orchestration_mode"] = "error"
        brief.trace["error"] = "r1_empty_required"
        brief.legacy_context = None
        brief.warnings.append("r1_empty_required")
        return brief

    if all_empty:
        if config.fallback_to_legacy_context:
            legacy = safe_legacy_context(
                build_context_fn=build_context_fn,
                enhanced_query=enhanced_query,
                news_title=brief.news_title,
                news_content=brief.news_content,
                key_concepts=brief.key_concepts,
                warnings=brief.warnings,
            )
            if legacy:
                brief.trace["orchestration_mode"] = "legacy_fallback"
                brief.legacy_context = legacy
                brief.warnings.append("all_slots_empty")
                return brief
            brief.trace["orchestration_mode"] = "error"
            brief.trace["error"] = default_error
            brief.legacy_context = None
            return brief
        brief.trace["orchestration_mode"] = "error"
        brief.trace["error"] = "all_slots_empty"
        brief.legacy_context = None
        return brief

    if r1_empty and config.require_r1:
        brief.warnings.append("r1_empty")
    brief.trace["orchestration_mode"] = "dialectical_v1"
    brief.mark_multi_stance()
    return brief


def _parallel_slots(
    *,
    brief: EvidenceBrief,
    config: DialecticalOrchestrationConfig,
    provider: Any,
    queries: dict[str, str],
    build_context_fn,
    enhanced_query: str | None,
) -> EvidenceBrief:
    specs = (
        ("r1", "core_self", config.r1_limit, True),
        ("r2", "influence_agree", config.r2_limit, False),
        ("r3", "influence_critical", config.r3_limit, False),
    )

    def _job(spec: tuple[str, str, int, bool]):
        slot_key, stance, limit, author_ok = spec
        started = time.perf_counter()
        try:
            items, step = retrieve_slot_with_fallback(
                provider=provider,
                query_text=queries[slot_key],
                stance_type=stance,
                limit=limit,
                widen_factor=config.widen_factor,
                allow_author_fallback=author_ok,
            )
            return slot_key, items, step, (time.perf_counter() - started) * 1000.0, None
        except Exception as error:  # noqa: BLE001
            return slot_key, [], "empty", (time.perf_counter() - started) * 1000.0, error

    # DO NOT call future.cancel() — does not stop running Qdrant work.
    # DO NOT call f.result()/f.exception() on not_done — may block forever.
    # shutdown(wait=False): do not block on abandoned slot threads at wall timeout.
    executor = ThreadPoolExecutor(max_workers=3)
    try:
        futures = {executor.submit(_job, spec): spec[0] for spec in specs}
        done, not_done = wait(
            futures.keys(),
            timeout=config.retrieve_wall_timeout_sec,
            return_when=ALL_COMPLETED,
        )
        if not_done:
            logger.warning(
                "retrieve_wall_timeout done_count=%s pending=%s",
                len(done),
                len(not_done),
            )
            brief.warnings.append("retrieve_wall_timeout")
            brief.r1_core_self = []
            brief.r2_influence_agree = []
            brief.r3_influence_critical = []
            brief.trace["fallback_steps"] = {
                "r1": "wall_timeout",
                "r2": "wall_timeout",
                "r3": "wall_timeout",
            }
            if config.fallback_to_legacy_context:
                legacy = safe_legacy_context(
                    build_context_fn=build_context_fn,
                    enhanced_query=enhanced_query,
                    news_title=brief.news_title,
                    news_content=brief.news_content,
                    key_concepts=brief.key_concepts,
                    warnings=brief.warnings,
                )
                if legacy:
                    brief.trace["orchestration_mode"] = "legacy_fallback"
                    brief.legacy_context = legacy
                    return brief
            brief.trace["orchestration_mode"] = "error"
            brief.trace["error"] = "retrieve_wall_timeout"
            brief.legacy_context = None
            return brief

        for future in done:
            slot_key, items, step, latency_ms, error = future.result()
            if error is not None:
                logger.warning("slot_provider_error slot=%s error=%s", slot_key, error)
                brief.warnings.append(f"{slot_key}_provider_error")
                step = "empty"
            brief.trace["slot_latency_ms"][slot_key] = round(latency_ms, 2)
            brief.trace["fallback_steps"][slot_key] = step
            if slot_key == "r1":
                brief.r1_core_self = items
            elif slot_key == "r2":
                brief.r2_influence_agree = items
            else:
                brief.r3_influence_critical = items
    finally:
        executor.shutdown(wait=False, cancel_futures=False)

    return _apply_empty_policies(
        brief=brief,
        config=config,
        build_context_fn=build_context_fn,
        enhanced_query=enhanced_query,
        default_error="provider_error",
    )

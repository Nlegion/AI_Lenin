"""Build EvidenceBrief with parallel slot retrieve and policy modes."""

from __future__ import annotations

from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
import logging
from pathlib import Path
import time
from typing import Any

from src.core.analysis.axes_extractor import extract_complementary_axes
from src.core.analysis.dialectical_config import DialecticalOrchestrationConfig
from src.core.analysis.evidence_brief import EvidenceBrief, truncate_query_for_trace
from src.core.analysis.semantic_core_config import SemanticCoreConfig, load_semantic_core_config
from src.core.analysis.semantic_integration import (
    apply_abstract_slot_queries,
    dialectical_uses_abstract,
    mark_fallback,
    maybe_route,
    timed_ms,
    trace_from_route,
)
from src.core.analysis.slot_retrieve import retrieve_slot_with_fallback

logger = logging.getLogger(__name__)

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
    semantic_config: SemanticCoreConfig | None = None,
    run_id: str | None = None,
    dialectical_enabled: bool = True,
) -> EvidenceBrief:
    warnings: list[str] = []
    semantic = semantic_config or load_semantic_core_config()
    route = maybe_route(
        news_title=news_title,
        news_content=news_content,
        config=semantic,
        run_id=run_id,
    )
    if route is not None:
        warnings.extend(route.warnings)

    use_axes = config.include_axes_in_query
    if dialectical_uses_abstract(
        semantic=semantic,
        dialectical_enabled=dialectical_enabled,
        route=route,
    ) and not semantic.include_axes_in_semantic_query:
        use_axes = False

    axes: list[str] = []
    if use_axes:
        axes, axis_warnings = extract_complementary_axes(
            news_title=news_title,
            news_content=news_content,
            key_concepts=key_concepts,
            axes_lemma_enabled=config.axes_lemma_enabled,
            taxonomy_path=taxonomy_path,
        )
        warnings.extend(axis_warnings)

    legacy_queries = {
        "r1": build_slot_query(
            news_title=news_title,
            news_content=news_content,
            key_concepts=key_concepts,
            axes=axes,
            modality_suffix=config.r1_modality_suffix,
            include_modality_suffix=config.include_modality_suffix,
            short_lead_chars=config.short_lead_chars,
            warnings=warnings,
        ),
        "r2": build_slot_query(
            news_title=news_title,
            news_content=news_content,
            key_concepts=key_concepts,
            axes=axes,
            modality_suffix=config.r2_modality_suffix,
            include_modality_suffix=config.include_modality_suffix,
            short_lead_chars=config.short_lead_chars,
            warnings=warnings,
        ),
        "r3": build_slot_query(
            news_title=news_title,
            news_content=news_content,
            key_concepts=key_concepts,
            axes=axes,
            modality_suffix=config.r3_modality_suffix,
            include_modality_suffix=config.include_modality_suffix,
            short_lead_chars=config.short_lead_chars,
            warnings=warnings,
        ),
    }

    used_abstract = dialectical_uses_abstract(
        semantic=semantic,
        dialectical_enabled=dialectical_enabled,
        route=route,
    )
    if used_abstract and route is not None:
        queries = apply_abstract_slot_queries(
            route=route,
            semantic=semantic,
            news_title=news_title,
            axes=axes,
            modality={
                "r1": config.r1_modality_suffix,
                "r2": config.r2_modality_suffix,
                "r3": config.r3_modality_suffix,
            },
            include_modality_suffix=config.include_modality_suffix,
        )
    else:
        queries = legacy_queries

    brief = EvidenceBrief(
        news_title=news_title,
        news_content=news_content,
        axes=axes,
        key_concepts=list(key_concepts),
        warnings=warnings,
        trace={
            "slot_queries": {
                "r1": truncate_query_for_trace(queries["r1"]),
                "r2": truncate_query_for_trace(queries["r2"]),
                "r3": truncate_query_for_trace(queries["r3"]),
            },
            "slot_latency_ms": {},
            "fallback_steps": {},
            "orchestration_mode": "dialectical_v1",
            "run_id": run_id,
            "semantic_fallback": False,
            "semantic_fallback_exhausted": False,
        },
    )
    if route is not None:
        brief.trace.update(trace_from_route(route))
        if route.synthesis_hints:
            brief.trace["synthesis_hints"] = list(route.synthesis_hints)

    if retrieval_provider is None:
        return _finalize_empty(
            brief=brief,
            config=config,
            build_context_fn=build_context_fn,
            enhanced_query=enhanced_query,
            reason="provider_unavailable",
        )

    brief = _parallel_slots(
        brief=brief,
        config=config,
        provider=retrieval_provider,
        queries=queries,
        build_context_fn=build_context_fn,
        enhanced_query=enhanced_query,
    )

    if (
        used_abstract
        and semantic.empty_r1_fallback_to_legacy_slot_query
        and not brief.r1_core_self
    ):
        started = time.perf_counter()
        brief.warnings.append("semantic_core_empty_r1_fallback")
        brief.r1_core_self = []
        brief.r2_influence_agree = []
        brief.r3_influence_critical = []
        brief.legacy_context = None
        brief.trace["orchestration_mode"] = "dialectical_v1"
        brief.trace.pop("error", None)
        brief.trace["slot_queries"] = {
            "r1": truncate_query_for_trace(legacy_queries["r1"]),
            "r2": truncate_query_for_trace(legacy_queries["r2"]),
            "r3": truncate_query_for_trace(legacy_queries["r3"]),
        }
        brief = _parallel_slots(
            brief=brief,
            config=config,
            provider=retrieval_provider,
            queries=legacy_queries,
            build_context_fn=build_context_fn,
            enhanced_query=enhanced_query,
        )
        elapsed = timed_ms(started)
        exhausted = (
            not brief.r1_core_self
            and not brief.r2_influence_agree
            and not brief.r3_influence_critical
            and not brief.legacy_context
        ) or brief.trace.get("orchestration_mode") == "error"
        mark_fallback(trace=brief.trace, elapsed_ms=elapsed, exhausted=bool(exhausted))
        if exhausted:
            brief.trace["orchestration_mode"] = "error"
            brief.trace["error"] = "semantic_fallback_exhausted"
            if "semantic_fallback_exhausted" not in brief.warnings:
                brief.warnings.append("semantic_fallback_exhausted")

    return brief


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
        _default_error=reason,
    )


def _apply_empty_policies(
    *,
    brief: EvidenceBrief,
    config: DialecticalOrchestrationConfig,
    build_context_fn,
    enhanced_query: str | None,
    _default_error: str,
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
            # Prefer legacy context over hard error so generation can still attempt analysis.
            if config.fallback_to_legacy_context and build_context_fn is not None:
                query = (enhanced_query or "").strip() or " ".join(
                    q for q in queries.values() if q
                ).strip()
                try:
                    brief.legacy_context = build_context_fn(query) if query else ""
                except Exception as error:  # noqa: BLE001
                    logger.warning("legacy_context_after_wall_timeout_failed error=%s", error)
                    brief.legacy_context = None
                if brief.legacy_context and str(brief.legacy_context).strip():
                    brief.trace["orchestration_mode"] = "legacy_fallback"
                    brief.trace["error"] = "retrieve_wall_timeout_legacy_fallback"
                    brief.warnings.append("retrieve_wall_timeout_legacy_fallback")
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
        _default_error="provider_error",
    )

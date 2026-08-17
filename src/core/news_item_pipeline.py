"""Per-news censor → generate → validate orchestration (extracted from NewsProcessor)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from src.core.generation.postprocess_clean import scrub_after_output_guard
from src.core.generation.publishability import (
    is_error_placeholder,
    is_publishable_analysis,
)
from src.core.ops.window_stats import WindowStats
from src.core.safety.pre_rag_censor import PreRagCensor
from src.core.safety.pre_rag_censor_types import CensorInput
from src.core.safety.yellow_audit import append_yellow_audit
from src.core.settings.analysis_defaults import REFUSAL_PHRASES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CensorAnnotateResult:
    """Stop/continue signal plus decision metadata for ops counters."""

    stop: str | None
    decision: str
    reason_codes: list[str]


def attach_censor_cache_callbacks(*, censor: PreRagCensor, repo: Any) -> None:
    """Bind request-local DB cache callbacks for one repository/session."""

    async def _load_cached(content_hash: str, config_hash: str):
        return await repo.get_censor_cached_decision(
            content_hash=content_hash,
            config_version_hash=config_hash,
        )

    async def _save_cached(
        content_hash: str, config_hash: str, model_hash: str, result
    ):
        confidence = dict(result.confidence)
        confidence["__context_hints__"] = [
            str(h.value if hasattr(h, "value") else h) for h in result.context_hints
        ]
        confidence["__needs_yellow_warning__"] = bool(result.needs_yellow_warning)
        await repo.upsert_censor_cached_decision(
            content_hash=content_hash,
            config_version_hash=config_hash,
            model_version_hash=model_hash,
            decision=result.decision,
            category=result.category,
            risk_tier=result.risk_tier,
            reason_codes=list(result.reason_codes),
            confidence=confidence,
        )

    async def _cleanup_cache(max_age_seconds: int):
        return await repo.cleanup_censor_cache(max_age_seconds=max_age_seconds)

    censor._load_cached_decision = _load_cached  # type: ignore[attr-defined]
    censor._save_cached_decision = _save_cached  # type: ignore[attr-defined]
    censor._cleanup_cached_decisions = _cleanup_cache  # type: ignore[attr-defined]


async def evaluate_and_annotate_news(
    *,
    news: Any,
    censor: PreRagCensor,
    classifier: Any,
    base_dir: Path,
) -> CensorAnnotateResult:
    """Run pre-RAG censor. stop='skip' when processing must stop."""
    censor_result = await censor.evaluate(
        CensorInput(
            news_id=str(news.id),
            title=news.title,
            body=news.content,
            source=news.source,
            metadata={"url": getattr(news, "url", "")},
        )
    )
    logger.info(
        "PreRagCensor decision news_id=%s decision=%s category=%s tier=%s codes=%s",
        news.id,
        censor_result.decision,
        censor_result.category,
        censor_result.risk_tier,
        ",".join(censor_result.reason_codes),
    )
    codes = list(censor_result.reason_codes)
    if censor_result.decision in {"hard_block", "skip"}:
        return CensorAnnotateResult(
            stop="skip",
            decision=str(censor_result.decision),
            reason_codes=codes,
        )
    if censor_result.decision == "review":
        logger.info(
            "Yellow risk publish mode enabled news_id=%s category=%s",
            news.id,
            censor_result.category,
        )
    setattr(news, "_risk_tier", censor_result.risk_tier)
    setattr(news, "_context_hints", [h.value for h in censor_result.context_hints])
    setattr(news, "_needs_yellow_warning", censor_result.needs_yellow_warning)
    if censor_result.risk_tier == "yellow":
        append_yellow_audit(
            base_dir=base_dir,
            item_id=str(news.id),
            title=news.title,
            content=news.content,
            risk_tier=censor_result.risk_tier,
            reason_codes=list(censor_result.reason_codes),
            decision=censor_result.decision,
        )
    should_analyze, reason = classifier.should_analyze(news.title, news.content)
    logger.info(
        "Classifier shadow news_id=%s should_analyze=%s reason=%s",
        news.id,
        should_analyze,
        reason,
    )
    return CensorAnnotateResult(
        stop=None,
        decision=str(censor_result.decision),
        reason_codes=codes,
    )


def _record_latency(stats: MutableMapping[str, int], analyzer: Any) -> None:
    meta = dict(getattr(analyzer, "last_pipeline_metadata", None) or {})
    if meta.get("cache_hit"):
        return
    latency = meta.get("latency_ms")
    if latency is None:
        return
    if isinstance(stats, WindowStats):
        stats.record_latency_ms(int(latency))


def _sync_circuit(stats: MutableMapping[str, int], analyzer: Any) -> None:
    if not hasattr(analyzer, "circuit_breaker"):
        return
    snap = analyzer.circuit_breaker.snapshot()
    if isinstance(stats, WindowStats):
        stats.sync_circuit_deltas(
            total_timeouts=int(snap.get("total_timeouts", 0) or 0),
            total_opens=int(snap.get("total_opens", 0) or 0),
        )
        return
    # Legacy dict: keep previous overwrite behavior for unit stubs.
    stats["generation_timeouts"] = snap.get("total_timeouts", 0)
    stats["circuit_opens"] = snap.get("total_opens", 0)


async def generate_and_persist_analysis(
    *,
    news: Any,
    analyzer: Any,
    news_guard: Any,
    validator: Any,
    repo: Any,
    stats: MutableMapping[str, int],
) -> None:
    """Generate, moderate, validate and optionally persist analysis."""
    logger.info("Генерация анализа для новости %s", news.id)
    analysis = await analyzer.generate_analysis(
        news.title,
        news.content,
        risk_tier=getattr(news, "_risk_tier", "green"),
        context_hints=getattr(news, "_context_hints", None),
        needs_yellow_warning=bool(getattr(news, "_needs_yellow_warning", False)),
    )
    _record_latency(stats, analyzer)
    if any(
        phrase in analysis.lower() for phrase in REFUSAL_PHRASES
    ) or is_error_placeholder(analysis):
        logger.info("Модель отказалась анализировать новость %s", news.id)
        await repo.mark_as_processed_without_analysis(news.id)
        stats["news_skipped"] = stats.get("news_skipped", 0) + 1
        if isinstance(stats, WindowStats):
            stats.record_skip_reasons(["refusal"])
        return

    logger.info("Сгенерирован анализ длиной %s символов", len(analysis))
    guard_blocked = False
    if news_guard is not None:
        guard_result = news_guard.guard_output(
            analysis=analysis,
            source_text=f"{news.title}\n{news.content}",
            risk_tier=getattr(news, "_risk_tier", "green"),
        )
        logger.info(
            "NewsGuard output news_id=%s blocked=%s codes=%s",
            news.id,
            guard_result.blocked,
            ",".join(guard_result.reason_codes),
        )
        if guard_result.blocked:
            guard_blocked = True
            stats["output_guard_blocked"] = stats.get("output_guard_blocked", 0) + 1
        analysis = scrub_after_output_guard(guard_result.moderated_text)

    validation = validator.validate_analysis(analysis, news.title)
    logger.info("Результат валидации: %s", validation)
    pipeline_meta = dict(getattr(analyzer, "last_pipeline_metadata", None) or {})
    if validation["is_valid"] and is_publishable_analysis(
        text=analysis, metadata=pipeline_meta
    ):
        await repo.save_analysis(news.id, analysis)
        stats["news_processed"] = stats.get("news_processed", 0) + 1
        logger.info(
            "Успешный анализ новости %s. Оценка: %.2f", news.id, validation["score"]
        )
        return

    if pipeline_meta.get("timeout_template_degrade") or pipeline_meta.get(
        "dialectical_outcome"
    ) in {
        "hold_review",
        "suppress",
    }:
        stats["degraded_held"] = stats.get("degraded_held", 0) + 1
    _sync_circuit(stats, analyzer)
    reasons = list(validation.get("reasons") or [])
    if guard_blocked and "output_guard_blocked" not in reasons:
        reasons.append("output_guard_blocked")
    logger.warning("Анализ новости %s отклонен: %s", news.id, ", ".join(reasons))
    await repo.mark_as_processed_without_analysis(news.id)
    stats["analyses_rejected"] = stats.get("analyses_rejected", 0) + 1
    if isinstance(stats, WindowStats):
        stats.record_reject_reasons(reasons)

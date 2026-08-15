"""Per-news censor → generate → validate orchestration (extracted from NewsProcessor)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.generation.postprocess_clean import scrub_after_output_guard
from src.core.generation.publishability import (
    is_error_placeholder,
    is_publishable_analysis,
)
from src.core.safety.pre_rag_censor import PreRagCensor
from src.core.safety.pre_rag_censor_types import CensorInput
from src.core.safety.yellow_audit import append_yellow_audit
from src.core.settings.analysis_defaults import REFUSAL_PHRASES

logger = logging.getLogger(__name__)


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
) -> str | None:
    """Run pre-RAG censor. Return 'skip' when processing must stop."""
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
    if censor_result.decision in {"hard_block", "skip"}:
        return "skip"
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
    return None


async def generate_and_persist_analysis(
    *,
    news: Any,
    analyzer: Any,
    news_guard: Any,
    validator: Any,
    repo: Any,
    stats: dict[str, int],
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
    if any(
        phrase in analysis.lower() for phrase in REFUSAL_PHRASES
    ) or is_error_placeholder(analysis):
        logger.info("Модель отказалась анализировать новость %s", news.id)
        await repo.mark_as_processed_without_analysis(news.id)
        stats["news_skipped"] += 1
        return

    logger.info("Сгенерирован анализ длиной %s символов", len(analysis))
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
        analysis = scrub_after_output_guard(guard_result.moderated_text)

    validation = validator.validate_analysis(analysis, news.title)
    logger.info("Результат валидации: %s", validation)
    pipeline_meta = dict(getattr(analyzer, "last_pipeline_metadata", None) or {})
    if validation["is_valid"] and is_publishable_analysis(
        text=analysis, metadata=pipeline_meta
    ):
        await repo.save_analysis(news.id, analysis)
        stats["news_processed"] += 1
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
    if hasattr(analyzer, "circuit_breaker"):
        snap = analyzer.circuit_breaker.snapshot()
        stats["generation_timeouts"] = snap.get("total_timeouts", 0)
        stats["circuit_opens"] = snap.get("total_opens", 0)
    logger.warning(
        "Анализ новости %s отклонен: %s", news.id, ", ".join(validation["reasons"])
    )
    await repo.mark_as_processed_without_analysis(news.id)
    stats["analyses_rejected"] += 1

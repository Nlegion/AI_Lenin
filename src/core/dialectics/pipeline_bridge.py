"""Bridge helpers: run dialectical engine from AnalysisGenerationPipeline."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from src.core.analysis.evidence_brief import EvidenceBrief
from src.core.dialectics.config import DialecticalMode, DialecticalReasoningConfig
from src.core.dialectics.engine import DialecticalEngine
from src.core.dialectics.judge import sample_judge
from src.core.dialectics.schemas import DialecticalRequest, DialecticalResult
from src.core.dialectics.shadow import should_sample_shadow, write_shadow_record
from src.core.llm.base import GenerationBackend
from src.core.generation.quality_hooks import apply_quality_post_generate
from src.core.generation.text_postprocess import clamp_answer_length
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

logger = logging.getLogger(__name__)


def brief_digest(brief: EvidenceBrief | None) -> str:
    if brief is None:
        return "none"
    parts: list[str] = []
    for item in [
        *brief.r1_core_self,
        *brief.r2_influence_agree,
        *brief.r3_influence_critical,
    ]:
        parts.append(f"{item.chunk_id}:{item.score:.4f}")
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def build_cache_suffix(
    *,
    mode: DialecticalMode,
    brief: EvidenceBrief | None,
    config: DialecticalReasoningConfig,
    model_name: str,
) -> str:
    return (
        f"{mode.value}:{brief_digest(brief)}:"
        f"{config.schema_version}:{config.engine_version}:{model_name}"
    )


async def run_reasoning_engine(
    *,
    backend: GenerationBackend,
    config: DialecticalReasoningConfig,
    news_title: str,
    news_content: str,
    brief: EvidenceBrief | None,
    enable_repair: bool,
    dialectical_applicable: bool = True,
) -> DialecticalResult:
    engine = DialecticalEngine(backend=backend, config=config)
    request = DialecticalRequest(
        news_title=news_title,
        news_content=news_content,
        dialectical_applicable=dialectical_applicable,
        fixture_mode=config.fixture_mode,
    )
    try:
        return await asyncio.wait_for(
            engine.analyze(request=request, brief=brief, enable_repair=enable_repair),
            timeout=config.global_timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning("dialectical_engine_timeout sec=%s", config.global_timeout_sec)
        return DialecticalResult(
            outcome="suppress",
            reason_codes=["timeout"],
            metadata={
                "fallback_to_legacy_on_timeout": config.fallback_to_legacy_on_timeout
            },
        )


async def maybe_attach_judge(
    *,
    backend: GenerationBackend,
    config: DialecticalReasoningConfig,
    result: DialecticalResult,
) -> DialecticalResult:
    if config.judge_sample_rate <= 0:
        return result
    import random

    if random.random() > config.judge_sample_rate:
        return result
    try:
        judged = await asyncio.wait_for(
            sample_judge(backend=backend, result=result),
            timeout=min(60.0, config.per_pass_timeout_sec),
        )
    except Exception as exc:  # noqa: BLE001
        result.metadata["judge_unavailable"] = str(exc)[:120]
        return result
    result.metadata["judge"] = judged
    # Hard validators win: judge may only downgrade.
    if result.outcome == "publish" and judged.get("fatal"):
        result.outcome = "hold_review"
        result.reason_codes = [*result.reason_codes, "judge_fatal"]
    return result


def apply_post_qc_for_reasoning(
    *,
    text: str,
    chunks: list[tuple[str, float, str]],
    candidates: list,
    brief: EvidenceBrief | None,
    config: QualityPostcheckConfig,
    news_text: str,
) -> tuple[str, dict[str, Any]]:
    """Quality post-pass with structure enforce skipped (engine owns structure)."""
    working, meta = apply_quality_post_generate(
        text=text,
        chunks=chunks,
        candidates=candidates,
        brief=brief,
        config=config,
        context_has_quotes=True,
        news_text=news_text,
        skip_structure_enforce=True,
    )
    before = working
    working, clamped = clamp_answer_length(working)
    if clamped:
        meta["answer_len_clamped_post_quality"] = True
    if working != before:
        meta["post_qc_modified"] = True
    # Soft revalidate structure labels after mutations.
    from src.core.generation.quality_hooks import _has_required_structure

    meta["structure_ok_after_post"] = _has_required_structure(working)
    if not meta["structure_ok_after_post"]:
        meta["post_qc_structure_broken"] = True
    return working, meta


def shadow_path(base_dir: Path) -> Path:
    return (
        base_dir
        / ".cursor"
        / "artifacts"
        / "quality"
        / "dialectical_reasoning_shadow.jsonl"
    )


def emit_shadow_if_sampled(
    *,
    base_dir: Path,
    config: DialecticalReasoningConfig,
    result: DialecticalResult,
    news_title: str,
    live_text: str,
) -> None:
    if not should_sample_shadow(config):
        return
    write_shadow_record(
        path=shadow_path(base_dir),
        result=result,
        news_title=news_title,
        mode=config.mode.value,
        live_text=live_text,
    )

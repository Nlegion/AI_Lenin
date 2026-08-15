"""Shared PreRagCensor wiring for live-news QA (batch + 24h)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from scripts.lib._quality_qa_io import QaItem
from scripts.lib._quality_qa_runtime import REFUSAL_FALLBACK
from src.core.safety.news_guard import NewsGuard
from src.core.safety.pre_rag_censor import PreRagCensor
from src.core.safety.pre_rag_censor_types import CensorInput, CensorResult
from src.core.safety.safety_gate import SafetyGate
from src.core.settings.censorship_runtime_config import (
    default_censorship_runtime_config_path,
    load_censorship_runtime_config,
)

logger = logging.getLogger("live_news_qa_batch")


@dataclass
class LiveGenerationContext:
    risk_tier: str = "green"
    context_hints: list[str] = field(default_factory=list)
    needs_yellow_warning: bool = False
    censor_decision: str = "allow"
    censor_reason_codes: list[str] = field(default_factory=list)


@dataclass
class LiveGateOutcome:
    blocked_row: dict[str, Any] | None = None
    generation: LiveGenerationContext | None = None


def build_live_pre_rag_censor(
    *,
    base_dir: Path,
    news_guard: NewsGuard | None = None,
    disable_unknown_forward: bool = True,
    enable_memory_cache: bool = True,
) -> PreRagCensor:
    """Mirror production/replay construction: SafetyGate + NewsGuard + runtime YAML."""
    cfg_path = default_censorship_runtime_config_path(base_dir)
    runtime_cfg = load_censorship_runtime_config(cfg_path)
    if disable_unknown_forward:
        runtime_cfg = replace(runtime_cfg, unknown_topic_to_skip_enabled=False)
    gate = SafetyGate.from_base_dir(base_dir)
    guard = news_guard or NewsGuard.from_file(path=base_dir / "config" / "news_guard.yaml")
    censor = PreRagCensor(
        safety_gate=gate,
        news_guard=guard,
        config=runtime_cfg,
        config_path=str(cfg_path),
    )
    if enable_memory_cache:
        _attach_memory_cache(censor)
    return censor


def _attach_memory_cache(censor: PreRagCensor) -> None:
    store: dict[tuple[str, str], dict[str, Any]] = {}

    async def _load(content_hash: str, config_hash: str) -> dict[str, Any] | None:
        return store.get((content_hash, config_hash))

    async def _save(
        content_hash: str,
        config_hash: str,
        model_hash: str,
        result: CensorResult,
    ) -> None:
        store[(content_hash, config_hash)] = {
            "decision": result.decision,
            "category": result.category,
            "risk_tier": result.risk_tier,
            "reason_codes": list(result.reason_codes),
            "confidence": dict(result.confidence),
            "context_hints": [str(h.value if hasattr(h, "value") else h) for h in result.context_hints],
            "needs_yellow_warning": bool(result.needs_yellow_warning),
            "model_version_hash": model_hash,
        }

    censor._load_cached_decision = _load  # noqa: SLF001
    censor._save_cached_decision = _save  # noqa: SLF001


def _map_skip_reason(decision: str) -> str:
    if decision == "hard_block":
        return "pre_deny"
    if decision == "skip":
        return "out_of_scope_skip"
    return "pre_quarantine"


async def apply_live_pre_rag_gate(
    *,
    censor: PreRagCensor,
    item: QaItem,
    row: dict[str, Any],
    strict_review: bool = False,
) -> LiveGateOutcome:
    """Apply production PreRagCensor before LLM.

    Parity mode (default): hard_block/skip -> reject; review -> yellow generation.
    Strict mode: review also rejects.
    """
    result = await censor.evaluate(
        CensorInput(
            news_id=str(item.id),
            title=item.title,
            body=item.content,
            source=item.source or "unknown",
        )
    )
    codes = list(result.reason_codes)
    hints = [str(h.value if hasattr(h, "value") else h) for h in result.context_hints]
    row["decision"] = result.decision
    row["censor_decision"] = result.decision
    row["censor_reason_codes"] = codes
    row["reason_codes"] = codes
    row["risk_tier"] = result.risk_tier
    row["context_hints"] = hints
    row["needs_yellow_warning"] = bool(result.needs_yellow_warning)

    if result.decision in {"hard_block", "skip"} or (
        strict_review and result.decision == "review"
    ):
        reason = _map_skip_reason(result.decision if result.decision != "review" else "review")
        message = (result.message or "").strip() or REFUSAL_FALLBACK
        row["status"] = "blocked"
        row["blocked"] = True
        row["skipped_llm"] = True
        row["skipped_llm_reason"] = reason
        row["answer"] = message
        row["prompt_builder"] = "pre_rag_censor"
        logger.info(
            "pre_rag_censor id=%s decision=%s reason=%s codes=%s",
            item.id,
            result.decision,
            reason,
            ",".join(codes),
        )
        return LiveGateOutcome(blocked_row=row)

    generation = LiveGenerationContext(
        risk_tier=result.risk_tier,
        context_hints=hints,
        needs_yellow_warning=bool(result.needs_yellow_warning) or result.decision == "review",
        censor_decision=result.decision,
        censor_reason_codes=codes,
    )
    if result.decision == "review":
        row["gate_yellow_pass"] = True
        logger.info("pre_rag_censor_yellow id=%s codes=%s", item.id, ",".join(codes))
    return LiveGateOutcome(generation=generation)

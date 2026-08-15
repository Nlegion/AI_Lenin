"""Pre-LLM gate policy for live-news QA (trusted unknown soft-pass)."""

from __future__ import annotations

import logging
from typing import Any

from scripts.lib._quality_qa_io import QaItem
from scripts.lib._quality_qa_runtime import REFUSAL_FALLBACK
from src.core.safety.drone_combat_guard import soft_pass_allowed
from src.core.safety.hotfix_flags import safety_flag_enabled
from src.core.safety.news_guard import NewsGuard

logger = logging.getLogger("live_news_qa_batch")

UNKNOWN_REASON = "no explicit allow topic matched"


def apply_live_pre_llm_gate(
    *,
    guard: NewsGuard,
    item: QaItem,
    row: dict[str, Any],
    unknown_as_allow: bool = True,
) -> dict[str, Any] | None:
    """Return blocked row for real safety hits; None to continue to LLM.

    When ``unknown_as_allow`` and gate would quarantine only because no allow-topic
    matched, proceed to LLM (row annotated with ``gate_soft_pass``) unless
    risk_tier is red or combat_adjacent_hit (quarantine / no-publish).
    """
    gate = guard.evaluate_input(
        title=item.title,
        content=item.content,
        source=item.source or "unknown",
    )
    if gate.decision == "allow":
        return None
    if (
        unknown_as_allow
        and gate.decision == "quarantine"
        and gate.reason == UNKNOWN_REASON
    ):
        blob = f"{item.title}\n{item.content}"
        if safety_flag_enabled("combat_adjacent_softpass_block"):
            allowed, block_codes = soft_pass_allowed(
                risk_tier=gate.risk_tier,
                text=blob,
            )
            if not allowed:
                row["status"] = "blocked"
                row["blocked"] = True
                row["skipped_llm"] = True
                row["skipped_llm_reason"] = "soft_pass_blocked_combat_adjacent"
                row["answer"] = (gate.message or "").strip() or REFUSAL_FALLBACK
                row["reason_codes"] = list(gate.reason_codes) + block_codes
                row["prompt_builder"] = "pre_llm_gate"
                row["gate_soft_pass_blocked"] = True
                logger.info(
                    "soft_pass_blocked id=%s codes=%s",
                    item.id,
                    ",".join(block_codes),
                )
                return row
        row["gate_soft_pass"] = "unknown_no_allow_topic"
        row["reason_codes"] = list(gate.reason_codes)
        logger.info("soft_pass id=%s reason=%s", item.id, gate.reason)
        return None
    if gate.decision == "deny":
        reason = "pre_deny"
    elif gate.decision == "skip":
        reason = "out_of_scope_skip"
    else:
        reason = "pre_quarantine"
    message = (gate.message or "").strip() or REFUSAL_FALLBACK
    row["status"] = "blocked"
    row["blocked"] = True
    row["skipped_llm"] = True
    row["skipped_llm_reason"] = reason
    row["answer"] = message
    row["reason_codes"] = list(gate.reason_codes)
    row["prompt_builder"] = "pre_llm_gate"
    logger.info(
        "pre_llm_gate id=%s decision=%s reason=%s codes=%s",
        item.id,
        gate.decision,
        reason,
        ",".join(gate.reason_codes),
    )
    return row

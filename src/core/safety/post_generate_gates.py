"""Apply non-mutating post-generate safety gates and build metadata fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.analysis.evidence_brief import EvidenceBrief
from src.core.safety.anachronism_gate import anachronism_gate
from src.core.safety.cliche_gate import cliche_gate
from src.core.safety.gate_warn_audit import append_gate_warn
from src.core.safety.groundedness_warn import news_groundedness
from src.core.safety.lacuna_hedge_gate import lacuna_hedge_gate
from src.core.safety.news_guard import NewsGuard, OutputGuardResult


def apply_post_generate_gates(
    *,
    text: str,
    brief: EvidenceBrief | None,
    news_title: str,
    news_content: str,
    news_guard: NewsGuard | None,
    post_filter: bool,
    warn_only_guard: bool,
    base_dir: Path,
    pipeline_id: str | None = None,
    risk_tier: str | None = None,
    yellow_block_patterns: list[str] | None = None,
) -> tuple[OutputGuardResult, dict[str, Any]]:
    """Run cliche → lacuna → anachronism → groundedness → NewsGuard. Gates do not modify ``text``."""
    r1_items = list(brief.r1_core_self) if brief is not None else []
    r1_text = "\n".join(item.text for item in r1_items)
    cliche_result = cliche_gate(
        analysis=text,
        brief_present=brief is not None,
        r1_text=r1_text,
        r1_count=len(r1_items),
    )
    if not cliche_result.skipped and cliche_result.reason_codes:
        append_gate_warn(
            gate="cliche",
            codes=cliche_result.reason_codes,
            analysis=text,
            base_dir=base_dir,
            pipeline_id=pipeline_id,
            r1_count=len(r1_items),
            r1_jaccard=cliche_result.r1_jaccard,
            lexicon_hits=cliche_result.lexicon_hits,
        )

    lacuna_result = lacuna_hedge_gate(analysis=text)
    if not lacuna_result.skipped and lacuna_result.reason_codes:
        append_gate_warn(
            gate="lacuna_hedge",
            codes=lacuna_result.reason_codes,
            analysis=text,
            base_dir=base_dir,
            pipeline_id=pipeline_id,
        )

    anachronism_result = anachronism_gate(analysis=text)
    if not anachronism_result.skipped and anachronism_result.reason_codes:
        append_gate_warn(
            gate="anachronism",
            codes=anachronism_result.reason_codes,
            analysis=text,
            base_dir=base_dir,
            pipeline_id=pipeline_id,
        )

    grounded = news_groundedness(
        analysis=text,
        news_title=news_title,
        news_content=news_content,
    )
    if not grounded.grounded:
        append_gate_warn(
            gate="groundedness",
            codes=["ungrounded_news_warn"],
            analysis=text,
            base_dir=base_dir,
            pipeline_id=pipeline_id,
        )

    if news_guard is not None and post_filter:
        from src.core.safety.risk_routing import RiskTier

        tier: RiskTier | None = None
        if risk_tier in {"red", "yellow", "green"}:
            tier = risk_tier  # type: ignore[assignment]
        guard_result = news_guard.guard_output(
            analysis=text,
            source_text=f"{news_title}\n{news_content}",
            warn_only=warn_only_guard,
            risk_tier=tier,
            extra_block_patterns=yellow_block_patterns,
        )
    else:
        guard_result = OutputGuardResult(
            blocked=False, moderated_text=text, reason_codes=[]
        )

    gate_metadata = {
        "cliche_gate": cliche_result.to_metadata(),
        "lacuna_hedge_gate": lacuna_result.to_metadata(),
        "anachronism_gate": anachronism_result.to_metadata(),
        "news_groundedness": grounded.to_metadata(),
        "guard_codes": list(guard_result.reason_codes),
        "risk_tier": risk_tier or "green",
    }
    return guard_result, gate_metadata

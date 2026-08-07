"""Typed contracts for SafetyGate (LLM-agnostic censorship interface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

Decision = Literal["allow", "deny", "quarantine", "skip"]
RiskTier = Literal["red", "yellow", "green"]
EnforceMode = Literal["old", "new"]


class SafetyHint(str, Enum):
    """Declarative prompt modifiers consumed by prompt_adapter."""

    AVOID_COMBAT_ESTIMATES = "avoid_combat_estimates"
    YELLOW_CONSTRAINED_ANALYSIS = "yellow_constrained_analysis"
    SEPARATE_FACT_OPINION = "separate_fact_opinion"
    NO_SPORT_REVOLUTION_ANALOGY = "no_sport_revolution_analogy"


@dataclass(frozen=True)
class GateContext:
    """Deterministic evaluation context (no hidden globals)."""

    title: str
    content: str
    source: str | None = None
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    item_id: str | None = None
    pipeline_id: str | None = None
    geo_hint: str | None = None
    config_version_hash: str = ""


@dataclass(frozen=True)
class RuleResult:
    hit: bool
    decision: Decision | None = None
    risk_tier: RiskTier | None = None
    reason: str = ""
    reason_codes: list[str] = field(default_factory=list)
    message: str = ""
    hints: list[SafetyHint] = field(default_factory=list)


@dataclass(frozen=True)
class GateDecision:
    decision: Decision
    risk_tier: RiskTier
    reason: str
    reason_codes: list[str]
    message: str = ""
    context_hints: list[SafetyHint] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    needs_yellow_warning: bool = False


@dataclass(frozen=True)
class ShadowCompareResult:
    enforced: GateDecision
    old_decision: GateDecision | None
    new_decision: GateDecision | None
    decision_match: bool
    reason_diff: list[str]
    config_version_hash: str

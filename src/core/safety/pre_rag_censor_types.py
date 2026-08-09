"""Contracts for standalone pre-RAG censorship module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from src.core.safety.safety_gate_types import SafetyHint

CensorDecision = Literal["allow", "hard_block", "review", "skip"]


@dataclass(frozen=True)
class CensorInput:
    news_id: str
    title: str
    body: str
    source: str | None = None
    published_at: datetime | None = None
    language_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizationMeta:
    content_hash: str
    normalized_text: str
    normalizer_version: str
    ru_ratio: float
    empty: bool
    duplicate_hit: bool
    duplicate_age_seconds: float | None


@dataclass(frozen=True)
class CensorResult:
    decision: CensorDecision
    category: str | None
    risk_tier: Literal["green", "yellow", "red"]
    reason_codes: list[str]
    reason: str
    message: str = ""
    confidence: dict[str, float] = field(default_factory=dict)
    context_hints: list[SafetyHint] = field(default_factory=list)
    needs_yellow_warning: bool = False
    audit: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

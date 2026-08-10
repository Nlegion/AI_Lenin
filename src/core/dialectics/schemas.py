"""Frozen contracts for dialectical reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PublishOutcome = Literal["publish", "hold_review", "suppress"]


@dataclass(frozen=True)
class PrincipleCard:
    principle_id: str
    title: str
    quote: str
    chunk_id: str
    stance_type: str
    source_path: str
    inferred: bool = False
    score: float = 0.0


@dataclass(frozen=True)
class CausalLink:
    cause: str
    condition: str
    effect: str
    theoretical_basis: str
    evidence_ids: list[str] = field(default_factory=list)
    principle_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class DialecticalTriad:
    thesis: str = ""
    antithesis: str = ""
    synthesis: str = ""
    thesis_from: str | None = None
    antithesis_from: str | None = None
    synthesis_basis: str | None = None


@dataclass
class QualityReport:
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class DialecticalRequest:
    news_title: str
    news_content: str
    dialectical_applicable: bool = True
    fixture_mode: bool = False


@dataclass
class DialecticalResult:
    outcome: PublishOutcome
    reason_codes: list[str] = field(default_factory=list)
    fact: str = ""
    triad: DialecticalTriad = field(default_factory=DialecticalTriad)
    mechanism_steps: list[str] = field(default_factory=list)
    conclusion: str = ""
    causal_links: list[CausalLink] = field(default_factory=list)
    used_principles: list[PrincipleCard] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    rendered_text: str = ""
    quality: QualityReport = field(default_factory=QualityReport)
    pass_timings_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    post_qc_modified: bool = False

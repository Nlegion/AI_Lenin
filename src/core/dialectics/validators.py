"""Hard validators for dialectical JSON results."""

from __future__ import annotations

import re
from typing import Any

from src.core.dialectics.schemas import (
    CausalLink,
    DialecticalResult,
    DialecticalTriad,
    PrincipleCard,
    QualityReport,
)

_BOILERPLATE = (
    "анализ опирается",
    "укрепление государства",
    "забота о гражданах",
    "положительное влияние",
    "объективная необходимость",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def build_result_from_payload(
    *,
    payload: dict[str, Any],
    cards: list[PrincipleCard],
) -> DialecticalResult:
    by_id = {c.principle_id: c for c in cards}
    used_ids = [str(x) for x in (payload.get("used_principle_ids") or []) if x]
    used = [by_id[i] for i in used_ids if i in by_id]
    links_raw = payload.get("causal_links") or []
    links: list[CausalLink] = []
    if isinstance(links_raw, list):
        for item in links_raw:
            if not isinstance(item, dict):
                continue
            links.append(
                CausalLink(
                    cause=str(item.get("cause") or ""),
                    condition=str(item.get("condition") or ""),
                    effect=str(item.get("effect") or ""),
                    theoretical_basis=str(item.get("theoretical_basis") or ""),
                    evidence_ids=[str(x) for x in (item.get("evidence_ids") or []) if x],
                    principle_ids=[str(x) for x in (item.get("principle_ids") or []) if x],
                    confidence=float(item.get("confidence") or 0.0),
                )
            )
    steps = payload.get("mechanism_steps") or []
    if not isinstance(steps, list):
        steps = []
    return DialecticalResult(
        outcome="hold_review",
        fact=str(payload.get("fact") or ""),
        triad=DialecticalTriad(
            thesis=str(payload.get("thesis") or ""),
            antithesis=str(payload.get("antithesis") or ""),
            synthesis=str(payload.get("synthesis") or ""),
            thesis_from=(str(payload["thesis_from"]) if payload.get("thesis_from") else None),
            antithesis_from=(
                str(payload["antithesis_from"]) if payload.get("antithesis_from") else None
            ),
            synthesis_basis=(
                str(payload["synthesis_basis"]) if payload.get("synthesis_basis") else None
            ),
        ),
        mechanism_steps=[str(s) for s in steps if s],
        conclusion=str(payload.get("conclusion") or ""),
        causal_links=links,
        used_principles=used,
        evidence_ids=[str(x) for x in (payload.get("evidence_ids") or []) if x],
        metadata={"r3_handling": str(payload.get("r3_handling") or "")},
    )


def validate_result(
    *,
    result: DialecticalResult,
    cards: list[PrincipleCard],
    has_r3: bool,
) -> QualityReport:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    card_ids = {c.principle_id for c in cards}
    chunk_ids = {c.chunk_id for c in cards}

    checks["has_fact"] = bool(result.fact.strip())
    checks["has_mechanism"] = bool(result.mechanism_steps) or bool(result.causal_links)
    checks["has_conclusion"] = bool(result.conclusion.strip())
    if not checks["has_fact"]:
        errors.append("missing_fact")
    if not checks["has_mechanism"]:
        errors.append("missing_mechanism")
    if not checks["has_conclusion"]:
        errors.append("missing_conclusion")

    unknown_principles = [
        p.principle_id for p in result.used_principles if p.principle_id not in card_ids
    ]
    # used_principles already filtered; also check payload ids via causal links
    for link in result.causal_links:
        for pid in link.principle_ids:
            if pid not in card_ids:
                unknown_principles.append(pid)
        for eid in link.evidence_ids:
            if eid not in chunk_ids:
                errors.append(f"unknown_evidence_id:{eid}")
    checks["ids_grounded"] = not unknown_principles and not any(
        e.startswith("unknown_evidence_id:") for e in errors
    )
    if unknown_principles:
        errors.append("unknown_principle_ids")

    blob = _norm(
        " ".join(
            [
                result.fact,
                result.triad.antithesis,
                result.conclusion,
                " ".join(result.mechanism_steps),
            ]
        )
    )
    boilerplate_hit = any(phrase in blob for phrase in _BOILERPLATE)
    checks["no_boilerplate"] = not boilerplate_hit
    if boilerplate_hit:
        errors.append("boilerplate_phrase")

    # Causal markers are a weak signal only — never a pass criterion alone.
    if result.mechanism_steps and len(" ".join(result.mechanism_steps)) < 40:
        warnings.append("thin_mechanism")

    if not has_r3:
        warnings.append("r3_absent")
        if (result.metadata or {}).get("r3_handling") not in {"r3_absent", "not_applicable", ""}:
            warnings.append("r3_handling_mismatch")

    # Opposing roles must not share the same chunk_id.
    t_from = result.triad.thesis_from
    a_from = result.triad.antithesis_from
    if t_from and a_from and t_from == a_from:
        errors.append("same_chunk_for_thesis_antithesis")
        checks["distinct_opposition_chunks"] = False
    else:
        checks["distinct_opposition_chunks"] = True

    if not result.used_principles and cards:
        errors.append("no_used_principles")
        checks["has_used_principles"] = False
    else:
        checks["has_used_principles"] = bool(result.used_principles) or not cards

    passed = not errors
    return QualityReport(passed=passed, errors=errors, warnings=warnings, checks=checks)

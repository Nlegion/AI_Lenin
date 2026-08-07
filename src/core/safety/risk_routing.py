"""Risk tier helpers and economy/yellow carve-outs for NewsGuard."""

from __future__ import annotations

from typing import Literal

from src.core.safety.pattern_match import pattern_hits

RiskTier = Literal["red", "yellow", "green"]

# Strong RF/ops markers — always red when hit (with combat/RF context rules in guard).
STRONG_MILITARY_MARKERS = (
    "вооруженные силы рф",
    "вооружённые силы рф",
    "вс рф",
    "специальная военная операция",
    "специальной военной операции",
    "мобилизац",
    "росгварди",
    "министерство обороны",
    "боевые действия",
    "армия россии",
)

DEFAULT_ECONOMY_POLICY_MARKERS = (
    "экономик",
    "страхов",
    "инфляц",
    "безработ",
    "санкц",
    "торговл",
    "экспорт",
    "импорт",
    "транзит",
    "инфраструктур",
    "железн",
    "железнодорож",
    "юкжд",
    "монопол",
    "энерго",
    "газпром",
    "нефтепровод",
    "трубопровод",
    "тариф",
    "бюджет",
    "зарплат",
    "забастовк",
    "профсоюз",
)


def map_decision_to_tier(decision: str) -> RiskTier:
    if decision == "deny":
        return "red"
    if decision in {"quarantine"}:
        return "yellow"
    if decision == "skip":
        return "green"  # soft-skip is not a combat risk tier
    return "green"


def has_economy_policy_marker(text: str, markers: list[str] | None = None) -> list[str]:
    patterns = list(markers) if markers else list(DEFAULT_ECONOMY_POLICY_MARKERS)
    return pattern_hits(text=text.lower(), patterns=patterns)


def strong_military_hits(text: str) -> list[str]:
    return pattern_hits(text=text.lower(), patterns=list(STRONG_MILITARY_MARKERS))


def yellow_economy_eligible(
    *,
    text: str,
    combat_hits: list[str],
    military_rf: bool,
    strong_military: list[str],
    other_red: list[str],
    economy_markers: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """yellow = economy && !combat && !RF context && !strong military && !other_red."""
    econ = has_economy_policy_marker(text, economy_markers)
    if not econ:
        return False, []
    if combat_hits or military_rf or strong_military or other_red:
        return False, econ
    return True, econ


def sport_intra_negative_hit(text: str, negatives: list[str]) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in negatives)


def policy_exception_markers(text: str) -> list[str]:
    """Positive markers that can lift soft-skip (labor/sanctions/protest/corruption)."""
    markers = (
        "забастовк",
        "профсоюз",
        "санкц",
        "против государств",
        "правительств",
        "госдума",
        "протест",
        "митинг",
        "коррупц",
        "финансирован",
        "бюджет",
        "политик",
    )
    return pattern_hits(text=text.lower(), patterns=list(markers))

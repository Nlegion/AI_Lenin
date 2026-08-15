"""Shared drone/air-raid deny and combat-adjacent soft-pass guards."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.safety.pattern_match import svo_token_hit, word_boundary_hit

DRONE_TOKENS = ("бпла", "беспилот", "дрон")
COMBAT_CO = (
    "атак",
    "удар",
    "сбит",
    "сбил",
    "опасность",
    "тревог",
    "обстрел",
    "ракет",
    "теракт",
)
CIVIL_NEGATIVES = (
    "кибер",
    "виру",
    "конкурент",
    "астм",
    "пешеход",
    "доставк",
    "агро",
    "сельхоз",
    "мониторинг",
    "шоу",
    "соревнован",
    "гражданск",
    "разработк",
    "предприят",
    "испытан",
    "грант",
    "учебн",
    "тренировочн",
    "планов",
    "технолог",
)
GEO_MARKERS = (
    "росси",
    " рф",
    "рф ",
    "украин",
    "киев",
    "харьков",
    "херсон",
    "одесс",
    "донбас",
    "луганск",
    "донецк",
    "крым",
    "белгород",
    "курск",
    "брянск",
    "пригранич",
    "област",
    "регион",
)
ADJACENT_NEGATIVES = (
    "учен",
    "тренировочн",
    "планов",
    "учебн",
    "разработк",
    "испытан",
)
EVAC_CIVIL = ("пожар", "наводнен", "разлив", "нефть", "чс", "мчс")
EXPLOSION_CIVIL = ("газ", "бытовой", "производственн", "техноген")
FRONT_FP = ("фронт работ", "народный фронт", "народного фронта")
ACTIVE_THREAT = ("опасность", "атак", "сбил", "сбит", "тревог")


@dataclass(frozen=True)
class GuardHit:
    hit: bool
    codes: list[str]


def _has_any(text: str, stems: tuple[str, ...] | list[str]) -> bool:
    return any(stem in text for stem in stems)


def _geo_hit(text: str) -> bool:
    return _has_any(text, GEO_MARKERS)


def _combat_co_present(text: str) -> bool:
    if svo_token_hit(lowered=text):
        return True
    return _has_any(text, ("обстрел", "удар", "теракт", "сбил", "сбит", "атак"))


def _drone_token_hit(text: str) -> bool:
    return _has_any(text, DRONE_TOKENS)


def _civil_negative_active(text: str) -> bool:
    if (
        "центр" in text
        and _drone_token_hit(text)
        and ("технолог" in text or "разработк" in text or "беспилотн" in text)
    ):
        return True
    return _has_any(text, CIVIL_NEGATIVES)


def _aftermath_clear(text: str) -> bool:
    if "сняли ограничения" in text or "отбой" in text:
        return not _has_any(text, ACTIVE_THREAT)
    return False


def drone_air_raid_hit(text: str) -> GuardHit:
    """Hard-deny signal for drone/air-raid with geo; combat co-token beats negatives."""
    lowered = text.lower()
    codes: list[str] = []
    if _aftermath_clear(lowered):
        return GuardHit(hit=False, codes=["drone:aftermath_clear"])

    drone = _drone_token_hit(lowered)
    co = _has_any(lowered, COMBAT_CO)
    geo = _geo_hit(lowered)

    phrase_patterns = (
        (r"опасност\w*\s+атак\w*\s+бпла", "phrase:danger_attack_bpla"),
        (r"беспилотн\w*\s+опасност", "phrase:uav_danger"),
        (r"ракетн\w*\s+опасност", "phrase:missile_danger"),
        (r"воздушн\w*\s+тревог", "phrase:air_raid_alarm"),
    )
    for pattern, code in phrase_patterns:
        if re.search(pattern, lowered) and geo:
            codes.append(code)

    if drone and co and geo:
        codes.append("drone_cooccurrence")
    if re.search(r"(сбил|сбит).{0,40}(бпла|беспилот|дрон)", lowered) and geo:
        codes.append("shot_down_drone")
    if (
        re.search(r"(бпла|беспилот|дрон).{0,40}(сбил|сбит|атак|удар|опасност)", lowered)
        and geo
    ):
        codes.append("drone_attack_context")

    if not codes:
        return GuardHit(hit=False, codes=[])

    if _civil_negative_active(lowered) and not _combat_co_present(lowered):
        return GuardHit(hit=False, codes=[*codes, "drone:civil_negative"])

    return GuardHit(hit=True, codes=codes)


def combat_adjacent_hit(text: str) -> GuardHit:
    """Soft-pass ban signal for combat-adjacent unknown topics."""
    lowered = text.lower()
    codes: list[str] = []
    geo = _geo_hit(lowered)

    drone = drone_air_raid_hit(text)
    if drone.hit:
        return GuardHit(hit=True, codes=[*drone.codes, "combat_adjacent:drone"])

    if svo_token_hit(lowered=lowered):
        codes.append("combat_adjacent:svo")

    if "обстрел" in lowered:
        codes.append("combat_adjacent:obstrel")

    if "фронт" in lowered and not any(fp in lowered for fp in FRONT_FP):
        if (
            word_boundary_hit(lowered, "фронт")
            or "фронте" in lowered
            or "фронту" in lowered
        ):
            codes.append("combat_adjacent:front")

    if "теракт" in lowered:
        codes.append("combat_adjacent:terror")

    if word_boundary_hit(lowered, "пво") and geo:
        codes.append("combat_adjacent:pvo")

    if re.search(r"(?<![а-яёa-z0-9])ракет(?!к)", lowered) and geo:
        codes.append("combat_adjacent:rocket")

    if re.search(r"боев\w*", lowered) and geo and "боевой листок" not in lowered:
        codes.append("combat_adjacent:combat")

    if "эвакуац" in lowered and geo and not _has_any(lowered, EVAC_CIVIL):
        codes.append("combat_adjacent:evac")

    if "взрыв" in lowered and not _has_any(lowered, EXPLOSION_CIVIL):
        siloviki = bool(re.search(r"силов\w*", lowered))
        if (
            "террорист" in lowered
            or (geo and siloviki)
            or (
                geo
                and any(
                    g in lowered
                    for g in ("украин", "харьков", "херсон", "одесс", "донбас")
                )
            )
        ):
            codes.append("combat_adjacent:explosion")

    if not codes:
        return GuardHit(hit=False, codes=[])

    training = _has_any(lowered, ADJACENT_NEGATIVES) and not _combat_co_present(lowered)
    if training:
        hard_codes = {
            "combat_adjacent:terror",
            "combat_adjacent:obstrel",
            "combat_adjacent:svo",
            "combat_adjacent:explosion",
        }
        if not any(c in hard_codes for c in codes):
            return GuardHit(
                hit=False, codes=[*codes, "combat_adjacent:training_negative"]
            )

    return GuardHit(hit=True, codes=codes)


def soft_pass_allowed(*, risk_tier: str, text: str) -> tuple[bool, list[str]]:
    """False when red or combat-adjacent — caller must quarantine/no-publish."""
    if risk_tier == "red":
        return False, ["soft_pass_block:risk_tier_red"]
    adjacent = combat_adjacent_hit(text)
    if adjacent.hit:
        return False, [f"soft_pass_block:{c}" for c in adjacent.codes]
    return True, []

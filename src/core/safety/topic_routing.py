"""Primary-topic / out-of-scope routing for NewsGuard."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Route = Literal["full", "skip", "none"]

POLICY_MARKERS = (
    "президент",
    "путин",
    "правительств",
    "санкц",
    "закон",
    "указ",
    "дума",
    "парламент",
    "митинг",
    "забастовк",
    "министерств",
    "кремл",
    "посольств",
    "дипломат",
    "выборы",
    "депутат",
    "протест",
    "чиновник",
)

KEY_MARKERS = (
    "президент",
    "путин",
    "правительств",
    "санкц",
    "закон",
    "кремл",
    "парламент",
    "дума",
    "выборы",
    "депутат",
    "митинг",
    "протест",
    "забастовк",
)

ACTION_VERBS = (
    "открыл",
    "подписал",
    "объявил",
    "ввёл",
    "ввел",
    "запретил",
    "утвердил",
    "поручил",
    "встретил",
    "провёл",
    "провел",
    "заявил",
    "сообщил",
    "назначил",
    "освободил",
    "выступил",
    "потребовал",
    "поддержал",
)

# Token-bound sport stems (avoid «паспорта»⊃«спорт»). Latin «sport» included.
SKIP_SPORT_TOKEN = ("спорт", "sport")
SKIP_SPORT_STEM = ("футбол", "теннис", "матч", "чемпионат", "олимп", "хоккей", "баскетбол")
SKIP_SCIENCE = ("палеонтолог", "динозавр", "археолог", "астроном", "погода", "прогноз погоды")
SKIP_CRIME = ("дтп", "авария", "уголовное дело", "кража", "ограбление")
SKIP_DISASTER = ("землетрясен", "наводнен", "ураган", "пожар в")
SOCIAL = ("здравоохран", "эпидем", "грипп", "медицин", "эколог", "школ", "образован", "университет")
LABOR_ECON = (
    "экономик",
    "инфляц",
    "безработ",
    "зарплат",
    "забастовк",
    "профсоюз",
    "санкц",
    "экспорт",
    "торговл",
    "страхов",
    "транзит",
    "инфраструктур",
    "железн",
    "железнодорож",
    "юкжд",
    "монопол",
    "энерго",
    "тариф",
    "бюджет",
)
DOPING_LIFT_FRAME = (
    "wada",
    "мок",
    "международн",
    "федерац",
    "министерств",
    "государств",
    "правительств",
    "президент",
    "санкц",
)
STATE_SANCTION_FRAME = (
    "правительств",
    "министр",
    "президент",
    "стран",
    "государств",
    "госдума",
    "посольств",
    "мид",
)


@dataclass(frozen=True)
class TopicRouteResult:
    route: Route
    primary: str
    reason_codes: list[str]


def _lead(title: str, content: str) -> str:
    first = re.split(r"(?<=[.!?])\s+", content.strip(), maxsplit=1)[0] if content.strip() else ""
    return f"{title}\n{first}".lower()


def _words(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def _marker_types(text: str, markers: tuple[str, ...]) -> set[str]:
    lowered = text.lower()
    return {m for m in markers if m in lowered}


def _abs_marker_count(text: str, markers: tuple[str, ...]) -> int:
    lowered = text.lower()
    total = 0
    for marker in markers:
        total += lowered.count(marker)
    return total


def title_lead_policy_full(title: str, content: str) -> list[str]:
    zone = _lead(title=title, content=content)
    types = _marker_types(zone, POLICY_MARKERS)
    keys = _marker_types(zone, KEY_MARKERS)
    verbs = [v for v in ACTION_VERBS if v in zone]
    if len(types) >= 2:
        return [f"title_lead_policy:{','.join(sorted(types))}"]
    if keys and verbs:
        return [f"title_lead_key_verb:{next(iter(keys))}+{verbs[0]}"]
    return []


def body_policy_override(content: str) -> list[str]:
    body = content.lower()
    words = _words(body)
    word_count = max(len(words), 1)
    abs_count = _abs_marker_count(body, POLICY_MARKERS)
    density = abs_count / (word_count * 0.01)
    unique = _marker_types(body, POLICY_MARKERS)
    keys = _marker_types(body, KEY_MARKERS)
    if density >= 1.0 and abs_count >= 2:
        return [f"body_density:{density:.2f}", f"abs:{abs_count}"]
    if len(unique) >= 3 and keys:
        return [f"body_distinct:{len(unique)}", f"key:{','.join(sorted(keys))}"]
    return []


def _sport_primary_hit(text: str) -> bool:
    """Token-prefix sport match: спортивный/спортсмен yes; паспорта no."""
    from src.core.safety.hotfix_flags import safety_flag_enabled

    lowered = text.lower()
    if safety_flag_enabled("sport_token_bound_enabled"):
        for tok in re.findall(r"[а-яёa-z0-9]+", lowered):
            if tok == "sport" or tok.startswith("спорт"):
                return True
        return any(stem in lowered for stem in SKIP_SPORT_STEM)
    return any(m in lowered for m in (*SKIP_SPORT_TOKEN, *SKIP_SPORT_STEM))


def classify_primary(title: str, content: str) -> str:
    text = f"{title}\n{content}".lower()
    if any(m in text for m in SOCIAL):
        return "social"
    if any(m in text for m in LABOR_ECON):
        return "labor_economy"
    if _sport_primary_hit(text):
        return "sport"
    if any(m in text for m in SKIP_SCIENCE):
        return "science"
    if any(m in text for m in SKIP_CRIME):
        return "crime"
    if any(m in text for m in SKIP_DISASTER):
        return "disaster"
    if any(m in text for m in ("политик", "геополит", "международ")):
        return "geopolitics"
    return "unknown"


def _sport_policy_lift(blob: str, positives: list[str]) -> bool:
    """Lift sport skip only with labor/protest or state/intl doping/sanctions frame."""
    lowered = blob.lower()
    if any(p.startswith("policy_exception") for p in positives):
        pass
    labor = any(m in lowered for m in ("забастовк", "протест", "митинг", "бойкот", "профсоюз"))
    if labor:
        return True
    if "госфинанс" in lowered or "финансирован" in lowered and "государств" in lowered:
        return True
    doping = any(m in lowered for m in ("допинг", "дисквалиф", "отстранен"))
    if doping and any(m in lowered for m in DOPING_LIFT_FRAME):
        return True
    if "санкц" in lowered and any(m in lowered for m in STATE_SANCTION_FRAME):
        return True
    # Non-sport policy markers from policy_exception_markers (corruption etc.)
    if any(m in lowered for m in ("коррупц", "политик", "бюджет")) and "санкц" not in lowered:
        return True
    if "санкц" in lowered:
        return any(m in lowered for m in STATE_SANCTION_FRAME)
    return bool(positives) and labor


def route_topic(
    *,
    title: str,
    content: str,
    sport_intra_negatives: list[str] | None = None,
) -> TopicRouteResult:
    from src.core.safety.risk_routing import policy_exception_markers, sport_intra_negative_hit

    codes = title_lead_policy_full(title=title, content=content)
    if codes:
        return TopicRouteResult(route="full", primary="policy", reason_codes=codes)
    body_codes = body_policy_override(content=content)
    if body_codes:
        return TopicRouteResult(route="full", primary="policy_body", reason_codes=body_codes)
    primary = classify_primary(title=title, content=content)
    blob = f"{title}\n{content}"
    if primary in {"sport", "science", "crime", "disaster"}:
        positives = policy_exception_markers(blob)
        if positives:
            lowered = blob.lower()
            state_frame = any(m in lowered for m in STATE_SANCTION_FRAME)
            if (
                primary == "sport"
                and sport_intra_negatives
                and sport_intra_negative_hit(blob, sport_intra_negatives)
                and not state_frame
            ):
                return TopicRouteResult(
                    route="skip",
                    primary=primary,
                    reason_codes=[f"out_of_scope:{primary}", "intra_domain_negative"],
                )
            if primary == "sport" and not _sport_policy_lift(blob, positives):
                return TopicRouteResult(
                    route="skip",
                    primary=primary,
                    reason_codes=[f"out_of_scope:{primary}", "sport_lift_insufficient"],
                )
            return TopicRouteResult(
                route="full",
                primary=primary,
                reason_codes=[f"policy_exception:{primary}", *positives],
            )
        return TopicRouteResult(route="skip", primary=primary, reason_codes=[f"out_of_scope:{primary}"])
    if primary in {"social", "labor_economy", "geopolitics"}:
        return TopicRouteResult(route="full", primary=primary, reason_codes=[f"primary:{primary}"])
    return TopicRouteResult(route="none", primary=primary, reason_codes=[])

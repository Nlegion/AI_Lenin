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

SKIP_SPORT = ("спорт", "футбол", "теннис", "матч", "чемпионат", "олимп", "хоккей", "баскетбол")
SKIP_SCIENCE = ("палеонтолог", "динозавр", "археолог", "астроном", "погода", "прогноз погоды")
SKIP_CRIME = ("дтп", "авария", "уголовное дело", "кража", "ограбление")
SKIP_DISASTER = ("землетрясен", "наводнен", "ураган", "пожар в")
SOCIAL = ("здравоохран", "эпидем", "грипп", "медицин", "эколог", "школ", "образован", "университет")
LABOR_ECON = ("экономик", "инфляц", "безработ", "зарплат", "забастовк", "профсоюз", "санкц", "экспорт", "торговл")


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


def classify_primary(title: str, content: str) -> str:
    text = f"{title}\n{content}".lower()
    if any(m in text for m in SOCIAL):
        return "social"
    if any(m in text for m in LABOR_ECON):
        return "labor_economy"
    if any(m in text for m in SKIP_SPORT):
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


def route_topic(*, title: str, content: str) -> TopicRouteResult:
    codes = title_lead_policy_full(title=title, content=content)
    if codes:
        return TopicRouteResult(route="full", primary="policy", reason_codes=codes)
    body_codes = body_policy_override(content=content)
    if body_codes:
        return TopicRouteResult(route="full", primary="policy_body", reason_codes=body_codes)
    primary = classify_primary(title=title, content=content)
    if primary in {"sport", "science", "crime", "disaster"}:
        return TopicRouteResult(route="skip", primary=primary, reason_codes=[f"out_of_scope:{primary}"])
    if primary in {"social", "labor_economy", "geopolitics"}:
        return TopicRouteResult(route="full", primary=primary, reason_codes=[f"primary:{primary}"])
    return TopicRouteResult(route="none", primary=primary, reason_codes=[])

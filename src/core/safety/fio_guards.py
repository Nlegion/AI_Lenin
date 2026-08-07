"""FIO false-positive guards (toponym / public-interest / roles) and charge keep."""

from __future__ import annotations

import re

from src.core.safety.hotfix_flags import safety_flag_enabled

FIO_PATTERN = re.compile(r"\b[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+\b")

TOPONYM_MARKERS = (
    "на земле",
    "в китае",
    "из китая",
    "китая",
    "представительниц",
    "остров",
    "полуостров",
    "архипелаг",
    "море",
    "реке",
    "горах",
    "тель-авив",
    "авиве",
    "израил",
    "washington",
    "вашингтон",
    "лондон",
    "берлин",
    "париж",
    "германи",
    "в германии",
    "молдав",
)

LENIN_ALLOWLIST = (
    "владимир ильич ленин",
    "в. и. ленин",
    "в.и. ленин",
)

CHARGE_MARKERS = (
    "обвинен",
    "подозрева",
    "уголов",
    "задержан",
    "арест",
    "следстви",
)

# Stem forms cover падежи (госдуму, фестиваля, молодежи).
PUBLIC_INTEREST_STEMS = (
    "госдум",
    "цик",
    "выбор",
    "голосован",
    "фестивал",
    "молодеж",
    "правительств",
    "президент",
    "министерств",
    "политик",
    "публичн",
    "экономик",
)

ROLE_MARKERS = (
    "министр",
    "губернатор",
    "посол",
    "руковод",
    "замруковод",
    "мэр",
    "судь",
    "депутат",
    "сенатор",
    "премьер",
    "председатель",
    "пресс-секрет",
    "пресс секрет",
    "официальн",
    "представител",
    "глава",
    "директор",
    "спецпредстав",
)

GOV_ORG_CONTEXT = (
    "мид",
    "правительств",
    "администрац",
    "ведомств",
    "цик",
    "госдум",
    "посольств",
    "министерств",
    "кремл",
    "федерац",
    "государств",
    "регион",
    "област",
    "мэр",
    "губернатор",
)

PRIVATE_VICTIM_MARKERS = (
    "мать погибш",
    "матери погибш",
    "родственник погибш",
    "семья погибш",
    "убитых россиян",
    "погибших россиян",
)


def fio_spans(text: str) -> list[re.Match[str]]:
    return list(FIO_PATTERN.finditer(text))


def is_lenin_attribution(span_text: str) -> bool:
    return span_text.lower() in LENIN_ALLOWLIST


def has_charge_context(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CHARGE_MARKERS)


def has_public_interest_context(text: str) -> bool:
    lowered = text.lower()
    if any(stem in lowered for stem in PUBLIC_INTEREST_STEMS):
        return True
    role = any(r in lowered for r in ROLE_MARKERS)
    gov = any(g in lowered for g in GOV_ORG_CONTEXT)
    return role and gov


def is_private_victim_context(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in PRIVATE_VICTIM_MARKERS)


def is_toponym_fio_false_positive(*, text: str, match: re.Match[str]) -> bool:
    """Heuristic: FIO-looking triple after toponym/preposition markers."""
    start = max(0, match.start() - 40)
    window = text[start : match.end()].lower()
    span_lower = match.group(0).lower()
    if is_lenin_attribution(span_text=span_lower):
        return True
    return any(marker in window for marker in TOPONYM_MARKERS)


def should_block_fio(*, text: str, matches: list[re.Match[str]]) -> list[str]:
    """Return reason codes for FIO denies; empty if all matches are FP or allowlisted."""
    if not matches:
        return []
    charge = has_charge_context(text=text)
    codes: list[str] = []
    carveout = safety_flag_enabled("fio_carveout_enabled")
    public = has_public_interest_context(text) if carveout else False
    if carveout and public and not charge:
        return []
    for match in matches:
        if is_lenin_attribution(span_text=match.group(0)):
            continue
        if not charge and is_toponym_fio_false_positive(text=text, match=match):
            continue
        codes.append(f"fio:{match.group(0)}")
    return codes

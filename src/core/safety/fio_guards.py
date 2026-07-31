"""FIO false-positive guards (toponym / preposition) and charge-context keep."""

from __future__ import annotations

import re

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


def fio_spans(text: str) -> list[re.Match[str]]:
    return list(FIO_PATTERN.finditer(text))


def is_lenin_attribution(span_text: str) -> bool:
    return span_text.lower() in LENIN_ALLOWLIST


def has_charge_context(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CHARGE_MARKERS)


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
    for match in matches:
        if is_lenin_attribution(span_text=match.group(0)):
            continue
        if not charge and is_toponym_fio_false_positive(text=text, match=match):
            continue
        codes.append(f"fio:{match.group(0)}")
    return codes

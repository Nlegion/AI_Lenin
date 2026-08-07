"""Lightweight fact vs opinion cue detection for prompt extras (no NER)."""

from __future__ import annotations

import re

_OPINION = re.compile(
    r"(?i)\b(?:по\s+мнению|считает|полагает)\b|"
    r"\b(?:эксперт|аналитик|обозреватель)\w*\s+\w+\s+(?:заявил|сообщил|считает|отметил)\b|"
    r"\b(?:эксперт|аналитик|обозреватель)\w*\s+(?:заявил|сообщил|считает)\b"
)
_EXPERT_SUBJECT = re.compile(r"(?i)\b(?:эксперт|аналитик|обозреватель)\w*\b")
_FACTUAL_VERB = re.compile(r"(?i)\b(?:сообщил|подтвердил|объявил)\b")


def needs_fact_opinion_extra(*, title: str, content: str) -> bool:
    """True when expert/opinion cues dominate; official reporting alone is false."""
    blob = f"{title}\n{content}"
    if _OPINION.search(blob):
        return True
    if _EXPERT_SUBJECT.search(blob) and _FACTUAL_VERB.search(blob):
        return True
    return False

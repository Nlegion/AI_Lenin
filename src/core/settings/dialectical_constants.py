"""Constants for dialectical R1–R3 orchestration and Jaccard metrics."""

from __future__ import annotations

# USER-FACING (RU). MVP hardcode; i18n later via config/locale if needed.
CONTEXT_UNAVAILABLE_MESSAGE = (
    "Не удалось получить контекст для анализа. Попробуйте позже."
)

# Function/content words only — NEVER proper names / domain terms.
JACCARD_STOPWORDS: frozenset[str] = frozenset(
    {
        "и",
        "в",
        "во",
        "на",
        "с",
        "со",
        "по",
        "для",
        "от",
        "до",
        "из",
        "к",
        "ко",
        "это",
        "как",
        "что",
        "чтобы",
        "или",
        "а",
        "но",
        "же",
        "бы",
        "ли",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "and",
        "or",
        "is",
        "are",
        "be",
        "as",
        "by",
        "with",
        "from",
        "at",
        "this",
        "that",
    }
)

# Validation-only — NEVER pass into tokenize() / jaccard().
JACCARD_DOMAIN_TERM_DENYLIST: frozenset[str] = frozenset(
    {
        "ленин",
        "капитал",
        "класс",
        "социализм",
        "коммунизм",
        "империализм",
        "пролетариат",
        "буржуазия",
        "диалектика",
        "революция",
    }
)

TRACE_QUERY_MAX_CHARS = 200
MIN_QDRANT_CLIENT_VERSION = "1.7.0"

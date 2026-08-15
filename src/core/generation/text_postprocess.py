"""Post-generation text helpers: strip markers, sentence trim, consecutive dedupe."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TRUNCATION_LEAK = re.compile(r"\n?\.\.\.\[truncated\]", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
# After sentence join, restore line breaks before triad labels so quality_hooks
# section regexes (^ / \\n) still see Факт / Механизм / Вывод as sections.
_TRIAD_SECTION_BREAK = re.compile(
    r"(?i)(?<=[.!?…])\s+(?=\*{0,2}(?:факт|механизм|вывод)\*{0,2}\s*:)"
)
# Single source of truth for publishable analysis length (Telegram framing ~4096).
MAX_FINAL_ANSWER_CHARS = 1800
_MAX_FINAL_ANSWER_CHARS = MAX_FINAL_ANSWER_CHARS  # backward-compatible alias


def restore_triad_section_breaks(text: str) -> str:
    """Re-insert newlines before triad labels flattened by sentence join."""
    if not text:
        return text
    return _TRIAD_SECTION_BREAK.sub("\n", text)


def strip_truncation_markers(text: str) -> str:
    if not text:
        return text
    return _TRUNCATION_LEAK.sub("", text).strip()


def truncate_to_last_complete_sentence(text: str) -> str:
    if not text:
        return text
    for index in range(len(text) - 1, -1, -1):
        if text[index] in ".!?…":
            return text[: index + 1]
    return text


def clamp_answer_length(
    text: str, *, max_chars: int = _MAX_FINAL_ANSWER_CHARS
) -> tuple[str, bool]:
    if not text or len(text) <= max_chars:
        return text, False
    candidate = text[:max_chars].rstrip()
    last_punct = max(
        candidate.rfind("."),
        candidate.rfind("!"),
        candidate.rfind("?"),
        candidate.rfind("…"),
    )
    if last_punct >= int(max_chars * 0.6):
        return candidate[: last_punct + 1], True
    return candidate, True


def dedupe_consecutive_sentences(text: str) -> tuple[str, dict[str, Any]]:
    """Drop exact consecutive sentence copies (eco_06-style loops)."""
    meta: dict[str, Any] = {"consecutive_repeat_removed": 0}
    if not text or not text.strip():
        return text, meta
    parts = _SENTENCE_SPLIT.split(text.strip())
    if len(parts) < 2:
        return text, meta
    kept: list[str] = []
    removed = 0
    previous_norm: str | None = None
    for part in parts:
        normalized = re.sub(r"\s+", " ", part.strip().casefold())
        if previous_norm is not None and normalized and normalized == previous_norm:
            removed += 1
            continue
        kept.append(part.strip())
        previous_norm = normalized if normalized else previous_norm
    meta["consecutive_repeat_removed"] = removed
    if removed:
        logger.info("consecutive_sentence_dedupe removed=%s", removed)
    return " ".join(item for item in kept if item), meta


def finalize_generated_text(text: str) -> tuple[str, dict[str, Any]]:
    cleaned = strip_truncation_markers(text)
    cleaned, dedupe_meta = dedupe_consecutive_sentences(cleaned)
    cleaned = restore_triad_section_breaks(cleaned)
    cleaned = truncate_to_last_complete_sentence(cleaned)
    cleaned, clamped = clamp_answer_length(cleaned)
    dedupe_meta["answer_len_clamped"] = clamped
    return cleaned, dedupe_meta

"""Identity pass for pipeline text after terminal post-guard scrub."""

from __future__ import annotations

from src.core.generation.publishability import is_error_placeholder

_EMPTY_FALLBACK = "Не удалось сгенерировать анализ."


def passthrough_pipeline_text(text: str) -> str:
    """Do not re-mutate post-guard output. Keep placeholders as-is."""
    if not text:
        return _EMPTY_FALLBACK
    if is_error_placeholder(text):
        return text
    return text

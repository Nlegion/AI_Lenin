"""Publishability checks for generated analysis text/metadata."""

from __future__ import annotations

from typing import Any

from src.core.settings.dialectical_constants import CONTEXT_UNAVAILABLE_MESSAGE

_ERROR_PLACEHOLDERS = frozenset(
    {
        "Ошибка анализа.",
        "Не удалось сгенерировать анализ.",
        CONTEXT_UNAVAILABLE_MESSAGE,
        "Анализ временно недоступен.",
    }
)


def is_error_placeholder(text: str | None) -> bool:
    cleaned = (text or "").strip()
    return cleaned in _ERROR_PLACEHOLDERS


def is_publishable_analysis(
    *,
    text: str | None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Return False for structure/hold/suppress/error outcomes."""
    meta = metadata or {}
    if is_error_placeholder(text):
        return False
    if bool(meta.get("structure_error")):
        return False
    if bool(meta.get("postprocess_hard_fail")):
        return False
    if str(meta.get("postprocess_status") or "") == "blocked":
        return False
    outcome = str(meta.get("dialectical_outcome") or "")
    if outcome in {"hold_review", "suppress"}:
        return False
    if str(meta.get("orchestration_mode") or "") == "error":
        return False
    grounded = meta.get("news_groundedness")
    if isinstance(grounded, dict) and grounded.get("ok") is False and bool(meta.get("structure_error")):
        return False
    return True

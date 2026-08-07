"""Typed soft-skip templates for out-of-scope primaries."""

from __future__ import annotations

from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

DEFAULT_SKIP = "Тема вне сферы марксистско-ленинского анализа новостей."


def skip_message_for_primary(primary: str, config: QualityPostcheckConfig | None = None) -> str:
    templates = (config.skip_templates if config is not None else {}) or {}
    return templates.get(primary) or templates.get("default") or DEFAULT_SKIP

"""Dialectical reasoning engine package (isolated from Telegram/processor)."""

from __future__ import annotations

from src.core.dialectics.config import (
    DialecticalMode,
    DialecticalReasoningConfig,
    load_dialectical_reasoning_config,
)
from src.core.dialectics.schemas import DialecticalResult, DialecticalRequest

__all__ = [
    "DialecticalMode",
    "DialecticalReasoningConfig",
    "DialecticalRequest",
    "DialecticalResult",
    "load_dialectical_reasoning_config",
]

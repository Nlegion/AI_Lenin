"""Degrade precedence and llama timeout circuit breaker."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Precedence (single pass, no double-fallback):
# 1) dialectical legacy_on_timeout
# 2) persona fallback (NewsGuard incidents) — separate module
# 3) template degrade (this module)
# 4) review/hold
SAFE_TEMPLATE = "Не удалось сформировать корректный анализ по данной новости."
DEGRADE_FLAG = "timeout_template_degrade"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    open_seconds: float = 60.0
    failures: int = 0
    opened_at: float | None = None
    total_timeouts: int = 0
    total_opens: int = 0

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_timeout(self) -> None:
        self.failures += 1
        self.total_timeouts += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()
            self.total_opens += 1
            logger.warning(
                "llama_circuit_open failures=%s open_seconds=%s",
                self.failures,
                self.open_seconds,
            )

    def allow_request(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.open_seconds:
            logger.info("llama_circuit_half_open")
            self.opened_at = None
            self.failures = 0
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "failures": self.failures,
            "opened": self.opened_at is not None,
            "total_timeouts": self.total_timeouts,
            "total_opens": self.total_opens,
        }


@dataclass
class DegradeResult:
    text: str
    publishable: bool
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


def template_degrade(*, reason: str) -> DegradeResult:
    """Stage-3 degrade: non-publishable safe template."""
    return DegradeResult(
        text=SAFE_TEMPLATE,
        publishable=False,
        stage="template_degrade",
        metadata={
            DEGRADE_FLAG: True,
            "degrade_reason": reason,
            "dialectical_outcome": "hold_review",
            "structure_error": True,
        },
    )

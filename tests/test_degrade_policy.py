from __future__ import annotations

import asyncio

from src.core.generation.degrade_policy import CircuitBreaker, template_degrade
from src.core.generation.errors import ErrorKind, classify_exception


def test_circuit_breaker_opens_after_timeouts() -> None:
    breaker = CircuitBreaker(failure_threshold=2, open_seconds=30.0)
    assert breaker.allow_request() is True
    breaker.record_timeout()
    assert breaker.allow_request() is True
    breaker.record_timeout()
    assert breaker.allow_request() is False
    snap = breaker.snapshot()
    assert snap["total_timeouts"] == 2
    assert snap["total_opens"] == 1


def test_template_degrade_is_non_publishable() -> None:
    result = template_degrade(reason="timeout")
    assert result.publishable is False
    assert result.metadata["timeout_template_degrade"] is True
    assert result.metadata["dialectical_outcome"] == "hold_review"


def test_classify_timeout_as_transient() -> None:
    assert classify_exception(asyncio.TimeoutError()) == ErrorKind.TRANSIENT
    assert classify_exception(ValueError("empty model content")) == ErrorKind.EMPTY

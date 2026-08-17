"""Unit tests for database UTC clock helper."""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.database.utc import utc_now


def test_utc_now_is_naive_and_near_wall_clock() -> None:
    before = datetime.now(UTC).replace(tzinfo=None)
    value = utc_now()
    after = datetime.now(UTC).replace(tzinfo=None)
    assert value.tzinfo is None
    assert before <= value <= after

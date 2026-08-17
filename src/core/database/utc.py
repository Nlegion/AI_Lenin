"""UTC clock helpers (SQLite DateTime columns store naive UTC)."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Current UTC as naive datetime for SQLite DateTime compatibility."""
    return datetime.now(UTC).replace(tzinfo=None)

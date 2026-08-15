"""Append-only JSONL helpers with non-raising OSError handling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def append_jsonl(path: Path, payload: dict[str, Any]) -> bool:
    """Append one JSON object as a line. Returns False on filesystem errors."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except OSError:
        logger.exception("jsonl_append_failed path=%s", path)
        return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows; skip malformed lines. Returns [] if missing or unreadable."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
    except OSError:
        logger.exception("jsonl_read_failed path=%s", path)
    return rows

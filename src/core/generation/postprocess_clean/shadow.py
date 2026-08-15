"""Read-only shadow compare for postprocess_clean vs live text."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.utils.jsonl import append_jsonl

logger = logging.getLogger(__name__)

_SHADOW_RELATIVE = (
    Path(".cursor") / "artifacts" / "quality" / "postprocess_clean_shadow.jsonl"
)


def shadow_log_path(base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else Path(__file__).resolve().parents[4]
    return root / _SHADOW_RELATIVE


def emit_shadow_record(
    *,
    phase: str,
    live_text: str,
    cloned_text: str,
    live_codes: list[str],
    cloned_codes: list[str],
    cloned_status: str,
    item_id: str | None = None,
    base_dir: Path | None = None,
) -> None:
    """Append one comparison row. Never mutates live_text."""
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "item_id": item_id,
        "text_equal": live_text == cloned_text,
        "codes_equal": list(live_codes) == list(cloned_codes),
        "live_codes": list(live_codes),
        "cloned_codes": list(cloned_codes),
        "cloned_status": cloned_status,
        "live_len": len(live_text),
        "cloned_len": len(cloned_text),
    }
    append_jsonl(shadow_log_path(base_dir=base_dir), payload)

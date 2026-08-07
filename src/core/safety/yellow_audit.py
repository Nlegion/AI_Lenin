"""Append yellow-tier decisions for offline weekly review."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def append_yellow_audit(
    *,
    base_dir: Path,
    item_id: str,
    title: str,
    content: str,
    risk_tier: str,
    reason_codes: list[str],
    decision: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if risk_tier != "yellow":
        return
    out_dir = base_dir / ".cursor" / "artifacts" / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"yellow_audit_{day}.jsonl"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "item_id": item_id,
        "text_hash": sha256(f"{title}\n{content}".encode("utf-8")).hexdigest()[:16],
        "risk_tier": risk_tier,
        "decision": decision,
        "reason_codes": reason_codes,
        **(extra or {}),
    }
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("yellow_audit_write_failed")

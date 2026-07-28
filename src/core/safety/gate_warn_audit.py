"""Append-only JSONL warn audit for post-generation gates."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.core.settings.gate_constants import GATE_WARN_AUDIT_PATH

logger = logging.getLogger(__name__)


def append_gate_warn(
    *,
    gate: str,
    codes: list[str],
    analysis: str,
    base_dir: Path | None = None,
    pipeline_id: str | None = None,
    r1_count: int | None = None,
    r1_jaccard: float | None = None,
    lexicon_hits: int | None = None,
) -> None:
    if not codes:
        return
    root = base_dir or Path.cwd()
    path = (root / GATE_WARN_AUDIT_PATH).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_id": pipeline_id or str(uuid4()),
        "gate": gate,
        "codes": list(codes),
        "r1_count": r1_count,
        "r1_jaccard": r1_jaccard,
        "lexicon_hits": lexicon_hits,
        "text_snippet": (analysis or "")[:200],
    }
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "gate_warn_audit",
            extra={"gate": gate, "codes": codes, "pipeline_id": record["pipeline_id"]},
        )
    except OSError:
        logger.exception("gate_warn_audit_write_failed")

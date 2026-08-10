"""Shadow JSONL logging for dialectical reasoning (never publishes)."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.dialectics.config import DialecticalReasoningConfig
from src.core.dialectics.schemas import DialecticalResult


def should_sample_shadow(config: DialecticalReasoningConfig, *, rng: random.Random | None = None) -> bool:
    rate = max(0.0, min(1.0, float(config.shadow_sample_rate)))
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    generator = rng or random.Random()
    return generator.random() < rate


def write_shadow_record(
    *,
    path: Path,
    result: DialecticalResult,
    news_title: str,
    mode: str,
    live_text: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "news_title": news_title,
        "outcome": result.outcome,
        "reason_codes": list(result.reason_codes),
        "rendered_text": result.rendered_text,
        "live_text": live_text,
        "quality_errors": list(result.quality.errors),
        "quality_warnings": list(result.quality.warnings),
        "pass_timings_ms": dict(result.pass_timings_ms),
        "post_qc_modified": result.post_qc_modified,
        "metadata": dict(result.metadata),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

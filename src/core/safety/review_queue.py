"""Review queue persistence for pre-RAG censorship decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.safety.pre_rag_censor_types import CensorInput, CensorResult


@dataclass(frozen=True)
class ReviewQueueItem:
    review_id: str
    news_id: str
    status: str
    created_at_utc: str
    title: str
    body: str
    source: str | None
    decision: str
    category: str | None
    risk_tier: str
    reason_codes: list[str]
    audit: dict[str, Any]
    final_decision: str | None = None
    resolved_at_utc: str | None = None
    provenance: str | None = None


def _queue_path(base_dir: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return base_dir / ".cursor" / "artifacts" / "quality" / f"review_queue_{day}.jsonl"


def enqueue_review_case(
    *,
    base_dir: Path,
    payload: CensorInput,
    result: CensorResult,
) -> str:
    review_id = f"{payload.news_id}:{int(datetime.now(timezone.utc).timestamp())}"
    item = ReviewQueueItem(
        review_id=review_id,
        news_id=payload.news_id,
        status="pending_review",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        title=payload.title,
        body=payload.body,
        source=payload.source,
        decision=result.decision,
        category=result.category,
        risk_tier=result.risk_tier,
        reason_codes=list(result.reason_codes),
        audit=dict(result.audit),
    )
    path = _queue_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    return review_id

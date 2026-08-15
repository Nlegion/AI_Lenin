"""JSON parse contract for dialectical LLM outputs (fence-aware, no XML)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

ParseStatus = Literal["parse_ok", "partial", "fail"]

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    data: dict[str, Any] | None
    error: str | None = None


def _try_load(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_candidates(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    candidates = [cleaned]
    for match in _FENCE.finditer(cleaned):
        body = match.group(1).strip()
        if body:
            candidates.insert(0, body)
    # Best-effort: outermost object span
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    return candidates


def parse_json_object(text: str) -> ParseResult:
    """Parse model output into a JSON object. Partial = missing keys handled by validators."""
    for candidate in _extract_candidates(text):
        data = _try_load(candidate)
        if data is not None:
            return ParseResult(status="parse_ok", data=data)
    # Truncated JSON: try closing braces once
    for candidate in _extract_candidates(text):
        if "{" not in candidate:
            continue
        repaired = candidate.rstrip()
        if not repaired.endswith("}"):
            repaired = repaired + "}"
        data = _try_load(repaired)
        if data is not None:
            return ParseResult(
                status="partial", data=data, error="truncated_json_repaired"
            )
    return ParseResult(status="fail", data=None, error="json_parse_failed")

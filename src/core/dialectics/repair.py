"""Field-level repair loop for dialectical JSON."""

from __future__ import annotations

import json
from typing import Any

from src.core.dialectics.config import DialecticalReasoningConfig
from src.core.dialectics.packing import PackedContext
from src.core.dialectics.prompts import REPAIR_USER_TEMPLATE, SYSTEM_PROMPT
from src.core.generation.base import GenerationRequest


def build_error_report(errors: list[str], *, max_chars: int) -> str:
    lines = [f"- {err}" for err in errors]
    text = "\n".join(lines)
    return text[:max_chars]


def build_repair_request(
    *,
    packed: PackedContext,
    previous_payload: dict[str, Any],
    errors: list[str],
    config: DialecticalReasoningConfig,
) -> GenerationRequest:
    report = build_error_report(errors, max_chars=config.max_error_report_chars)
    previous = json.dumps(previous_payload, ensure_ascii=False)[:2000]
    user = REPAIR_USER_TEMPLATE.format(
        news_block=packed.news_block,
        principle_block=packed.principle_block,
        previous_json=previous,
        error_report=report,
    )
    return GenerationRequest(
        system_prompt=SYSTEM_PROMPT,
        user_content=user,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )


def error_set_progressed(previous: list[str], current: list[str]) -> bool:
    return len(set(current)) < len(set(previous))

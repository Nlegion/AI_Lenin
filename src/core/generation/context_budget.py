"""Approx token budget + chunk-first / chars-second context shrink."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CTX_SIZE = 4096
WARN_RATIO = 0.9
CHUNK_STEPS = (7, 5, 3)
CHARS_FLOOR = 2000
CHARS_SHRINK_FACTOR = 0.8


def approx_tokens(text: str) -> int:
    """Cheap estimate: ~4 chars per token for mixed RU/EN."""
    return max(1, (len(text) + 3) // 4) if text else 0


@dataclass
class BudgetState:
    max_context_chars: int
    max_context_chunks: int
    ctx_size: int = DEFAULT_CTX_SIZE
    max_tokens: int = 512
    shrink_steps: list[dict[str, Any]] = field(default_factory=list)


def budget_over_limit(*, prompt_text: str, state: BudgetState) -> bool:
    used = approx_tokens(prompt_text) + int(state.max_tokens)
    return used >= int(state.ctx_size * WARN_RATIO)


def log_budget(*, prompt_text: str, state: BudgetState) -> int:
    prompt_tokens = approx_tokens(prompt_text)
    total = prompt_tokens + int(state.max_tokens)
    limit = int(state.ctx_size * WARN_RATIO)
    if total >= limit:
        logger.warning(
            "token_budget_near_limit approx_prompt_tokens=%s max_tokens=%s total=%s limit=%s "
            "chunks=%s max_context_chars=%s",
            prompt_tokens,
            state.max_tokens,
            total,
            limit,
            state.max_context_chunks,
            state.max_context_chars,
        )
    else:
        logger.info(
            "token_budget_ok approx_prompt_tokens=%s max_tokens=%s total=%s",
            prompt_tokens,
            state.max_tokens,
            total,
        )
    return prompt_tokens


def next_chunk_cap(current: int) -> int | None:
    for value in CHUNK_STEPS:
        if current > value:
            return value
    return None


def shrink_budget(state: BudgetState) -> bool:
    """Apply one shrink step. Returns False when floors reached."""
    next_chunks = next_chunk_cap(state.max_context_chunks)
    if next_chunks is not None:
        state.max_context_chunks = next_chunks
        step = {
            "context_shrink_step": "chunks",
            "chunks": state.max_context_chunks,
            "max_context_chars": state.max_context_chars,
        }
        state.shrink_steps.append(step)
        logger.info("context_shrink %s", step)
        return True
    if state.max_context_chars > CHARS_FLOOR:
        shrunk = max(CHARS_FLOOR, int(state.max_context_chars * CHARS_SHRINK_FACTOR))
        if shrunk >= state.max_context_chars:
            return False
        state.max_context_chars = shrunk
        step = {
            "context_shrink_step": "chars",
            "chunks": state.max_context_chunks,
            "max_context_chars": state.max_context_chars,
        }
        state.shrink_steps.append(step)
        logger.info("context_shrink %s", step)
        return True
    return False


def re_split_blocks(context: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", context.strip())
    if len(blocks) >= 2:
        return blocks
    return re.split(r"(?=\n(?:R[123]|\[fallback))", context)


def clip_context_by_chunks(context: str, *, max_chunks: int) -> str:
    """Heuristic: split on blank lines / section markers; keep top-N segments."""
    if max_chunks <= 0 or not context.strip():
        return context
    parts = [block.strip() for block in re_split_blocks(context) if block.strip()]
    if len(parts) <= max_chunks:
        return context.strip()
    return "\n\n".join(parts[:max_chunks])

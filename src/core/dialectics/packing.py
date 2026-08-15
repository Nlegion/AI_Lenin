"""Token packing for dialectical prompts under ctx_size budget."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.dialectics.config import DialecticalReasoningConfig
from src.core.dialectics.schemas import PrincipleCard
from src.core.generation.context_budget import approx_tokens


@dataclass
class PackedContext:
    news_block: str
    principle_block: str
    dropped_chunk_ids: list[str]
    approx_prompt_tokens: int


def _format_card(card: PrincipleCard) -> str:
    return (
        f"- id={card.principle_id} stance={card.stance_type} "
        f"chunk={card.chunk_id}\n  «{card.quote}»"
    )


def pack_reasoning_context(
    *,
    news_title: str,
    news_fact: str,
    cards: list[PrincipleCard],
    system_prompt: str,
    user_prefix: str,
    config: DialecticalReasoningConfig,
    error_report: str = "",
) -> PackedContext:
    """Priority: output reserve → news → R1 → R2/R3 → errors."""
    limit = int(config.ctx_size * config.ctx_margin_ratio) - int(config.max_tokens_out)
    news_block = f"TITLE:\n{news_title.strip()}\nFACT:\n{news_fact.strip()}"
    fixed = (
        approx_tokens(system_prompt)
        + approx_tokens(user_prefix)
        + approx_tokens(news_block)
    )
    if error_report:
        fixed += approx_tokens(error_report[: config.max_error_report_chars])

    r1 = [c for c in cards if c.stance_type == "core_self"]
    other = [c for c in cards if c.stance_type != "core_self"]
    selected: list[PrincipleCard] = []
    dropped: list[str] = []
    used = fixed
    for card in [*r1, *other]:
        block = _format_card(card)
        cost = approx_tokens(block)
        if used + cost > limit and selected:
            dropped.append(card.chunk_id)
            continue
        selected.append(card)
        used += cost

    principle_block = (
        "\n".join(_format_card(c) for c in selected) if selected else "(none)"
    )
    return PackedContext(
        news_block=news_block,
        principle_block=principle_block,
        dropped_chunk_ids=dropped,
        approx_prompt_tokens=used,
    )

"""Render DialecticalResult to Fact/Mechanism/Conclusion within length budget."""

from __future__ import annotations

from src.core.dialectics.config import DialecticalReasoningConfig
from src.core.dialectics.schemas import DialecticalResult
from src.core.generation.text_postprocess import clamp_answer_length


def render_analysis_text(
    *,
    result: DialecticalResult,
    config: DialecticalReasoningConfig,
) -> tuple[str, bool]:
    principle_hint = ""
    if result.used_principles:
        quote = result.used_principles[0].quote
        if len(quote) > 120:
            quote = quote[:117].rstrip() + "…"
        principle_hint = f" Опора: «{quote}»."

    if result.mechanism_steps:
        mechanism = " ".join(result.mechanism_steps)
    elif result.causal_links:
        link = result.causal_links[0]
        mechanism = (
            f"Из-за {link.cause} при условии {link.condition} происходит {link.effect}."
        )
    elif result.triad.antithesis:
        mechanism = result.triad.antithesis
    else:
        mechanism = "Недостаточно теоретического обоснования в контексте."

    mechanism = f"{mechanism}{principle_hint}".strip()
    fact = result.fact or result.triad.thesis or "Факт новости не извлечён."
    conclusion = result.conclusion or result.triad.synthesis or "Вывод ограничен доступным контекстом."

    text = f"Факт: {fact}\nМеханизм: {mechanism}\nВывод: {conclusion}".strip()
    clamped, changed = clamp_answer_length(text, max_chars=config.max_rendered_chars)
    return clamped, changed

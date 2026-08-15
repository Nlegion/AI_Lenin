"""DeepSeek-only chat request builders (does not modify llama prompt_adapter)."""

from __future__ import annotations

from src.core.generation.prompt_adapter import (
    ANACHRONISM_PROMPT_RULE,
    DIALECTICAL_SYSTEM_EXTRA,
    FACT_OPINION_EXTRA,
    FALLBACK_LEGACY_FRAME,
    FALLBACK_LEGACY_MARKER,
    GIGACHAT_SYSTEM_PROMPT,
    INTERNAL_LABEL_BAN_EXTRA,
    SOCIAL_EMPTY_R1_EXTRA,
    SOCIAL_FACT_ANCHOR_EXTRA,
    SPORT_ANALOGY_BAN_EXTRA,
    YELLOW_CONSTRAINT_EXTRA,
    _hint_block,
    _hints_extras,
    _truncate_context,
)
from src.core.llm.base import GenerationRequest

DEEPSEEK_STRUCTURE_EXTRA = """
Структура ответа обязательна:
1) «Факт: ...» — одно конкретное проверяемое предложение из новости.
2) «Механизм: ...» — 2–4 предложения; утверждающий ленинский анализ
   (без hedging вроде «может рассматриваться»).
3) «Вывод: ...» — 1–2 предложения, категоричный итог.
   Не пиши «Условие применимости».
"""

DEEPSEEK_STRICT_QUOTE_EXTRA = """
Цитатное правило (strict):
- В блоке «Механизм» включи минимум одну прямую цитату из списка допустимых R1-цитат.
- Выбери цитату, которая по смыслу связывается с фактом новости (не вставляй
  случайный фрагмент ради кавычек).
- Сразу после цитаты одной короткой фразой поясни связку: какой факт новости
  объясняется принципом из цитаты.
- Оформляй цитату в кавычках «…» дословно и целиком (не обрывай середину фразы).
- Не оставляй обрывков вроде «:.» / «,.» / фраз без сказуемого.
- Не выдумывай том, страницу, название работы и не приписывай Ленину фразы вне списка.
- Либо есть дословная цитата из списка, либо её нет — нельзя совмещать оба варианта.
- Если ни одна допустимая цитата не подходит: напиши без кавычек фразу
  В предоставленном контексте подходящей цитаты нет
  и строй анализ на принципах БЕЗ любых кавычек «» / "" и без вводных
  «как писал Ленин» / «Ленин отмечал».
"""

DEEPSEEK_PRINCIPLES_EXTRA = """
В релевантном R1 нет пригодных цитат. Запрещено выдумывать цитаты, кавычки и том/стр.
Дай краткий анализ через принципы: Факт → Механизм → Вывод.
Не пиши «Условие применимости». Не используй кавычки вообще.
"""

DEEPSEEK_QUOTE_FEEDBACK = (
    "Нужна ровно одна дословная цитата из допустимых R1-цитат в кавычках «…», "
    "либо фраза без кавычек: В предоставленном контексте подходящей цитаты нет. "
    "Нельзя совмещать отказ от цитаты с кавычками. Не обрывай цитату."
)


def _deepseek_system(
    *,
    dialectical: bool,
    usable_excerpts: bool,
    feedback: list[str] | None,
    synthesis_hints: list[str] | None,
    hint_only: bool,
    social_primary: bool,
    empty_r1: bool,
    fact_opinion: bool,
    risk_tier: str,
    sport_primary: bool,
    context_hints: list[str] | None,
) -> str:
    system_prompt = GIGACHAT_SYSTEM_PROMPT
    if dialectical:
        system_prompt += "\n" + DIALECTICAL_SYSTEM_EXTRA
    if feedback:
        system_prompt += "\nУчти замечания:\n" + "\n".join(
            f"- {item}" for item in feedback
        )
    system_prompt += "\n" + ANACHRONISM_PROMPT_RULE
    hint = _hint_block(synthesis_hints=synthesis_hints, hint_only=hint_only)
    if hint:
        system_prompt += "\n" + hint
    system_prompt += "\n" + DEEPSEEK_STRUCTURE_EXTRA
    system_prompt += "\n" + INTERNAL_LABEL_BAN_EXTRA
    if usable_excerpts:
        system_prompt += "\n" + DEEPSEEK_STRICT_QUOTE_EXTRA
    else:
        system_prompt += "\n" + DEEPSEEK_PRINCIPLES_EXTRA
    if social_primary and empty_r1:
        system_prompt += "\n" + SOCIAL_EMPTY_R1_EXTRA
    elif social_primary:
        system_prompt += "\n" + SOCIAL_FACT_ANCHOR_EXTRA
    hint_block = _hints_extras(context_hints)
    if hint_block:
        system_prompt += "\n" + hint_block
    else:
        if fact_opinion:
            system_prompt += "\n" + FACT_OPINION_EXTRA
        if risk_tier == "yellow":
            system_prompt += "\n" + YELLOW_CONSTRAINT_EXTRA
        if sport_primary:
            system_prompt += "\n" + SPORT_ANALOGY_BAN_EXTRA
    return system_prompt


def _compose_context(
    *,
    context: str,
    max_context_chars: int,
    excerpts_block: str,
    legacy_fallback: bool,
) -> str:
    context_block = _truncate_context(context=context, max_chars=max_context_chars)
    if legacy_fallback and context_block:
        context_block = (
            f"{FALLBACK_LEGACY_MARKER}\n{FALLBACK_LEGACY_FRAME}\n{context_block}"
        )
    if excerpts_block:
        if context_block:
            return f"{excerpts_block}\n\n{context_block}"
        return excerpts_block
    return context_block


def build_deepseek_chat_request(
    *,
    news_title: str,
    news_content: str,
    context: str,
    max_context_chars: int,
    excerpts_block: str = "",
    usable_excerpts: bool = False,
    feedback: list[str] | None = None,
    synthesis_hints: list[str] | None = None,
    hint_only: bool = False,
    legacy_fallback: bool = False,
    social_primary: bool = False,
    empty_r1: bool = False,
    fact_opinion: bool = False,
    risk_tier: str = "green",
    sport_primary: bool = False,
    context_hints: list[str] | None = None,
) -> GenerationRequest:
    context_block = _compose_context(
        context=context,
        max_context_chars=max_context_chars,
        excerpts_block=excerpts_block,
        legacy_fallback=legacy_fallback,
    )
    system_prompt = _deepseek_system(
        dialectical=False,
        usable_excerpts=usable_excerpts,
        feedback=feedback,
        synthesis_hints=synthesis_hints,
        hint_only=hint_only,
        social_primary=social_primary,
        empty_r1=empty_r1,
        fact_opinion=fact_opinion,
        risk_tier=risk_tier,
        sport_primary=sport_primary,
        context_hints=context_hints,
    )
    user_content = (
        f"Новость: {news_title}\n{news_content[:400]}\n\n"
        f"Контекст RAG (цитаты и provenance):\n{context_block}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return GenerationRequest(
        system_prompt=system_prompt,
        user_content=user_content,
        messages=messages,
    )


def build_deepseek_dialectical_chat_request(
    *,
    news_title: str,
    news_content: str,
    context: str,
    max_context_chars: int,
    excerpts_block: str = "",
    usable_excerpts: bool = False,
    feedback: list[str] | None = None,
    synthesis_hints: list[str] | None = None,
    hint_only: bool = False,
    social_primary: bool = False,
    empty_r1: bool = False,
    fact_opinion: bool = False,
    risk_tier: str = "green",
    sport_primary: bool = False,
    context_hints: list[str] | None = None,
) -> GenerationRequest:
    context_block = _compose_context(
        context=context,
        max_context_chars=max_context_chars,
        excerpts_block=excerpts_block,
        legacy_fallback=False,
    )
    system_prompt = _deepseek_system(
        dialectical=True,
        usable_excerpts=usable_excerpts,
        feedback=feedback,
        synthesis_hints=synthesis_hints,
        hint_only=hint_only,
        social_primary=social_primary,
        empty_r1=empty_r1,
        fact_opinion=fact_opinion,
        risk_tier=risk_tier,
        sport_primary=sport_primary,
        context_hints=context_hints,
    )
    user_content = (
        f"Новость: {news_title}\n{news_content[:400]}\n\n"
        f"Доказательная база (не выдумывай вне этих блоков):\n{context_block}\n\n"
        "Задача: краткий анализ в стиле Ленина, связывающий новость с R1 "
        "и при необходимости с опорой/критикой из R2/R3."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return GenerationRequest(
        system_prompt=system_prompt,
        user_content=user_content,
        messages=messages,
    )

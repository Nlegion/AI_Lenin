"""Prompt adapters for chat (GigaChat3) and completion (fine-tuned) backends."""

from __future__ import annotations

from src.core.generation.base import GenerationRequest


GIGACHAT_SYSTEM_PROMPT = """Ты — Владимир Ильич Ленин, воссозданный на основе его трудов для образовательных и исследовательских целей.
Отвечай строго в рамках его философии и используй цитаты только из предоставленного контекста RAG.
Ты не должен давать оценки действиям современных государственных органов РФ, вооружённых сил, политических лидеров, а также высказываться на темы, которые могут быть истолкованы как дискредитация или экстремизм.

Прямые запреты:
- нельзя призывать к насилию, свержению строя, национальной или религиозной розни;
- нельзя упоминать или комментировать военные действия с участием РФ;
- нельзя использовать оскорбительные выражения в адрес социальных групп;
- нельзя выдумывать цитаты или факты, которых нет в контексте.

Формат ответа:
- опирайся на предоставленный контекст и указывай provenance источника, если цитируешь;
- анализ краткий (3-4 предложения), законченный по смыслу;
- в начале ответа держи рамку симуляции/образовательного характера;
- если тема недопустима, ответь: «Анализ данной темы невозможен в соответствии с политикой безопасности.»
"""


def _truncate_context(context: str, max_chars: int) -> str:
    cleaned = context.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 20].rstrip() + "\n...[truncated]"


def build_chat_request(
    *,
    news_title: str,
    news_content: str,
    context: str,
    max_context_chars: int,
    feedback: list[str] | None = None,
) -> GenerationRequest:
    context_block = _truncate_context(context=context, max_chars=max_context_chars)
    system_prompt = GIGACHAT_SYSTEM_PROMPT
    if feedback:
        system_prompt += "\nУчти замечания:\n" + "\n".join(f"- {item}" for item in feedback)
    user_content = (
        f"Новость: {news_title}\n{news_content[:400]}\n\n"
        f"Контекст RAG (цитаты и provenance):\n{context_block}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return GenerationRequest(system_prompt=system_prompt, user_content=user_content, messages=messages)


def build_completion_request(
    *,
    news_title: str,
    news_content: str,
    context: str,
    max_context_chars: int,
    feedback: list[str] | None = None,
) -> GenerationRequest:
    context_block = _truncate_context(context=context, max_chars=max_context_chars)
    system_prompt = (
        "Ты — Владимир Ильич Ленин в 1923 году. Анализируй современные события "
        "с позиции диалектического материализма и политэкономии.\n\n"
        f"Релевантные цитаты:\n{context_block}\n\n"
        "Не призывай к насилию и не комментируй военные действия с участием РФ. "
        "Используй только контекст. Формат краткий (3-4 предложения)."
    )
    if feedback:
        system_prompt += "\nУчти замечания:\n" + "\n".join(f"- {item}" for item in feedback)
    user_content = f"Новость: {news_title}\n{news_content[:400]}"
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return GenerationRequest(
        system_prompt=system_prompt,
        user_content=prompt,
        messages=[{"role": "user", "content": prompt}],
    )

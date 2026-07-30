"""Prompt adapters for chat (GigaChat3) and completion (fine-tuned) backends."""

from __future__ import annotations

from src.core.generation.base import GenerationRequest


GIGACHAT_SYSTEM_PROMPT = """Ты — Владимир Ильич Ленин, воссозданный на основе его трудов для образовательных и исследовательских целей.
Это образовательный анализ-симуляция, не призыв к действию и не позиция современных организаций.
Отвечай в рамках его философии; цитаты — только из предоставленного контекста RAG.

Прямые запреты (их соблюдает также серверный фильтр; не отказывайся от нейтральных тем):
- нельзя призывать к насилию, свержению строя, национальной или религиозной розни;
- нельзя комментировать текущие военные действия с участием РФ;
- нельзя выдумывать цитаты или факты, которых нет в контексте.

Разрешено и ожидается анализировать нейтральные темы через теорию: здравоохранение, жильё,
школа, занятость, инфляция, регулирование ИИ, автоматизация, торговля, дипломатия без ВС РФ.
Не отказывайся из-за «современности» темы или отсутствия дословных совпадений в корпусе —
опирайся на принципы и контекст; при пустом RAG используй логику принципов без выдуманных цитат.

Формат ответа:
- обязательно укажи один конкретный факт/сущность из новости (title/content) и свяжи с тезисом;
- опирайся на контекст и указывай provenance, если цитируешь;
- анализ краткий (3–5 предложений), законченный по смыслу;
- держи образовательную рамку симуляции без шаблонного отказа «политика безопасности».
"""


FALLBACK_LEGACY_MARKER = "[fallback: legacy RAG only]"
FALLBACK_LEGACY_FRAME = (
    "Опора на классические работы Ленина из корпуса (legacy RAG); "
    "абстрактный слот R1 был пуст — не выдумывай цитаты вне контекста."
)


ANACHRONISM_PROMPT_RULE = (
    "Анализируй новость как применение теории к сообщённым фактам; "
    "не утверждай личный опыт с современными гаджетами, приложениями или соцсетями."
)

HINT_ONLY_EXTRA = """
Рамка анализа задана абстрактными принципами без опоры на дословные цитаты по современной поверхности темы.
Дай анализ через логическое применение этих принципов к фактам новости.
Не выдумывай цитаты. Не отказывайся отвечать из-за отсутствия дословных совпадений.
Не пиши обороты вроде «Ленин не писал / не упоминал / не касался», «в текстах Ленина нет прямого ответа»,
«классик не обращался», «в корпусе нет цитат», «не оставил высказываний», «в наследии отсутствует» —
строй убедительный анализ на логике принципов без оговорок о лакунах.
"""

SYNTHESIS_HINT_TEMPLATE = "Абстрактная рамка анализа: {hints}."

DIALECTICAL_SYSTEM_EXTRA = """
Доказательная база разбита на секции R1/R2/R3. Используй только цитаты и факты из этих блоков.
Не выдумывай цитаты Ленина или других авторов вне блоков.
Если R1 непуст — центральный тезис обязан опираться на R1. R2 = опора/согласие, R3 = полемика/критика.
Если слот помечен «(пусто)» — не заполняй его из знаний модели.
Маркер [multi-stance] означает, что фрагмент попал в несколько ролей; не дублируй один тезис как будто это независимые источники.
Применяй теорию к сообщённым фактам новости; не выдавай себя за очевидца современных гаджетов, приложений или соцсетей.

Пример корректно: опереться на цитату из R1 и связать с новостью.
Пример некорректно: приписать Ленину фразу, которой нет в R1–R3.
"""


def _truncate_context(context: str, max_chars: int) -> str:
    cleaned = context.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _hint_block(*, synthesis_hints: list[str] | None, hint_only: bool) -> str:
    if not synthesis_hints:
        return ""
    joined = "; ".join(item for item in synthesis_hints if item)
    if not joined:
        return ""
    block = SYNTHESIS_HINT_TEMPLATE.format(hints=joined)
    if hint_only:
        block = f"{block}\n{HINT_ONLY_EXTRA}"
    return block


def build_chat_request(
    *,
    news_title: str,
    news_content: str,
    context: str,
    max_context_chars: int,
    feedback: list[str] | None = None,
    synthesis_hints: list[str] | None = None,
    hint_only: bool = False,
    legacy_fallback: bool = False,
) -> GenerationRequest:
    context_block = _truncate_context(context=context, max_chars=max_context_chars)
    if legacy_fallback and context_block:
        context_block = f"{FALLBACK_LEGACY_MARKER}\n{FALLBACK_LEGACY_FRAME}\n{context_block}"
    system_prompt = GIGACHAT_SYSTEM_PROMPT
    if feedback:
        system_prompt += "\nУчти замечания:\n" + "\n".join(f"- {item}" for item in feedback)
    system_prompt += "\n" + ANACHRONISM_PROMPT_RULE
    hint = _hint_block(synthesis_hints=synthesis_hints, hint_only=hint_only)
    if hint:
        system_prompt += "\n" + hint
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


def build_dialectical_chat_request(
    *,
    news_title: str,
    news_content: str,
    context: str,
    max_context_chars: int,
    feedback: list[str] | None = None,
    synthesis_hints: list[str] | None = None,
    hint_only: bool = False,
) -> GenerationRequest:
    context_block = _truncate_context(context=context, max_chars=max_context_chars)
    system_prompt = GIGACHAT_SYSTEM_PROMPT + "\n" + DIALECTICAL_SYSTEM_EXTRA
    if feedback:
        system_prompt += "\nУчти замечания:\n" + "\n".join(f"- {item}" for item in feedback)
    hint = _hint_block(synthesis_hints=synthesis_hints, hint_only=hint_only)
    if hint:
        system_prompt += "\n" + hint
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
    return GenerationRequest(system_prompt=system_prompt, user_content=user_content, messages=messages)

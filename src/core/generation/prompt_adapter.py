"""Prompt adapters for GigaChat3 chat completions."""

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
- анализ краткий, но структурированный: Факт -> Механизм -> Вывод;
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

QUOTE_REQUIRE_EXTRA = """
В контексте есть релевантные цитаты. Требуется ≥1 цитата из контекста, наиболее связанная с фактами новости.
Укажи один конкретный факт новости. Не выдумывай том/стр, если их нет в meta чанка.
"""

PRINCIPLES_NO_QUOTE_EXTRA = """
В релевантном контексте нет подходящих цитат. Запрещено выдумывать цитаты, кавычки и том/стр.
Дай развёрнутый анализ через принципы, опираясь на факты новости (3–5 предложений).
Не пиши «Ленин сказал/писал» без дословной опоры в контексте.
Используй числа, названные сущности и официальные действия из заголовка/текста новости.
Не отказывайся анализировать из‑за отсутствия цитат; не используй шаблонный отказ.
Не повторяй длинный юридический дисклеймер в начале ответа.
"""

FACT_OPINION_EXTRA = """
В новости есть экспертные или оценочные суждения. Явно отделяй факты события от мнений:
пиши «эксперт заявляет / считает, что…». Строй анализ на сообщённых фактах, а не на интерпретации эксперта.
"""

YELLOW_CONSTRAINT_EXTRA = """
Режим ограниченного анализа (yellow): разбирай экономические и политические отношения.
Запрещено комментировать боевые действия, тактику, перемещения войск, потери, призывы к насилию.
"""

SPORT_ANALOGY_BAN_EXTRA = """
Не проводи прямых аналогий спортивных событий с революционной борьбой, «пробуждением масс»
или свержением строя, если в новости нет прямого политического протеста или классового конфликта.
"""

ALLOWLIST_QUOTE_HEADER = "Допустимые цитаты из RAG (используй только их дословно, без выдуманных том/стр):"

SOCIAL_FACT_ANCHOR_EXTRA = """
Тема социальная (здравоохранение/экология/образование). Анализ основан на фактах новости;
ленинские концепции — только при прямой связи с этими фактами. Избегай шаблонов без опоры в новости.
Если фактов мало — ответ короче.
"""

SOCIAL_EMPTY_R1_EXTRA = """
Начни с 1–2 конкретных фактов новости. Не используй кавычки и не ссылайся на Ленина как на цитату,
если в предоставленном контексте нет прямых цитат. Без общих фраз без опоры в новости.
"""

HINT_ONLY_EXTRA = """
Рамка анализа задана абстрактными принципами без опоры на дословные цитаты по современной поверхности темы.
Дай анализ через логическое применение этих принципов к фактам новости.
Не выдумывай цитаты. Не отказывайся отвечать из-за отсутствия дословных совпадений.
Не пиши обороты вроде «Ленин не писал / не упоминал / не касался», «в текстах Ленина нет прямого ответа»,
«классик не обращался», «в корпусе нет цитат», «не оставил высказываний», «в наследии отсутствует» —
строй убедительный анализ на логике принципов без оговорок о лакунах.
"""

ANALYSIS_STRUCTURE_EXTRA = """
Структура ответа обязательна:
1) «Факт: ...» — один конкретный проверяемый факт из новости.
2) «Механизм: ...» — ленинская объяснительная рамка (без выдуманных цитат).
3) «Вывод: ...» — причинно-следственный итог и ограничение/условие применимости.
"""

INTERNAL_LABEL_BAN_EXTRA = """
Запрещено выводить внутренние технические маркеры и служебные токены:
- R1, R2, R3 как служебные ссылки;
- [multi-stance] и похожие теги;
- специальные токены вида <|im_start|>, <|im_end|>.
"""

SYNTHESIS_HINT_TEMPLATE = "Абстрактная рамка анализа: {hints}."


def _hints_extras(context_hints: list[str] | None) -> str:
    """Map typed SafetyHint values to prompt paragraphs (ALLOW/green => empty)."""
    if not context_hints:
        return ""
    parts: list[str] = []
    hints = {str(h) for h in context_hints}
    if "yellow_constrained_analysis" in hints or "avoid_combat_estimates" in hints:
        parts.append(YELLOW_CONSTRAINT_EXTRA)
    if "separate_fact_opinion" in hints:
        parts.append(FACT_OPINION_EXTRA)
    if "no_sport_revolution_analogy" in hints:
        parts.append(SPORT_ANALOGY_BAN_EXTRA)
    return "\n".join(parts)


def _mode_extras(
    *,
    quote_mode: str,
    social_primary: bool,
    empty_r1: bool,
    fact_opinion: bool = False,
    risk_tier: str = "green",
    sport_primary: bool = False,
    context_hints: list[str] | None = None,
) -> str:
    parts: list[str] = []
    parts.append(ANALYSIS_STRUCTURE_EXTRA)
    parts.append(INTERNAL_LABEL_BAN_EXTRA)
    if quote_mode == "quote":
        parts.append(QUOTE_REQUIRE_EXTRA)
    else:
        parts.append(PRINCIPLES_NO_QUOTE_EXTRA)
    if social_primary and empty_r1:
        parts.append(SOCIAL_EMPTY_R1_EXTRA)
    elif social_primary:
        parts.append(SOCIAL_FACT_ANCHOR_EXTRA)
    # Prefer typed hints from SafetyGate; fall back to legacy risk_tier/flags.
    hint_block = _hints_extras(context_hints)
    if hint_block:
        parts.append(hint_block)
    else:
        if fact_opinion:
            parts.append(FACT_OPINION_EXTRA)
        if risk_tier == "yellow":
            parts.append(YELLOW_CONSTRAINT_EXTRA)
        if sport_primary:
            parts.append(SPORT_ANALOGY_BAN_EXTRA)
    return "\n".join(parts)


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
    quote_mode: str = "principles",
    social_primary: bool = False,
    empty_r1: bool = False,
    fact_opinion: bool = False,
    risk_tier: str = "green",
    sport_primary: bool = False,
    allowlist_quotes: list[str] | None = None,
    context_hints: list[str] | None = None,
) -> GenerationRequest:
    context_block = _truncate_context(context=context, max_chars=max_context_chars)
    if legacy_fallback and context_block:
        context_block = f"{FALLBACK_LEGACY_MARKER}\n{FALLBACK_LEGACY_FRAME}\n{context_block}"
    if allowlist_quotes and quote_mode == "quote":
        bullets = "\n".join(f"- «{q}»" for q in allowlist_quotes if q)
        context_block = f"{ALLOWLIST_QUOTE_HEADER}\n{bullets}\n\n{context_block}"
    system_prompt = GIGACHAT_SYSTEM_PROMPT
    if feedback:
        system_prompt += "\nУчти замечания:\n" + "\n".join(f"- {item}" for item in feedback)
    system_prompt += "\n" + ANACHRONISM_PROMPT_RULE
    hint = _hint_block(synthesis_hints=synthesis_hints, hint_only=hint_only)
    if hint:
        system_prompt += "\n" + hint
    system_prompt += "\n" + _mode_extras(
        quote_mode=quote_mode,
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
    return GenerationRequest(system_prompt=system_prompt, user_content=user_content, messages=messages)


def build_dialectical_chat_request(
    *,
    news_title: str,
    news_content: str,
    context: str,
    max_context_chars: int,
    feedback: list[str] | None = None,
    synthesis_hints: list[str] | None = None,
    hint_only: bool = False,
    quote_mode: str = "principles",
    social_primary: bool = False,
    empty_r1: bool = False,
    fact_opinion: bool = False,
    risk_tier: str = "green",
    sport_primary: bool = False,
    allowlist_quotes: list[str] | None = None,
    context_hints: list[str] | None = None,
) -> GenerationRequest:
    context_block = _truncate_context(context=context, max_chars=max_context_chars)
    if allowlist_quotes and quote_mode == "quote":
        bullets = "\n".join(f"- «{q}»" for q in allowlist_quotes if q)
        context_block = f"{ALLOWLIST_QUOTE_HEADER}\n{bullets}\n\n{context_block}"
    system_prompt = GIGACHAT_SYSTEM_PROMPT + "\n" + DIALECTICAL_SYSTEM_EXTRA
    if feedback:
        system_prompt += "\nУчти замечания:\n" + "\n".join(f"- {item}" for item in feedback)
    hint = _hint_block(synthesis_hints=synthesis_hints, hint_only=hint_only)
    if hint:
        system_prompt += "\n" + hint
    system_prompt += "\n" + _mode_extras(
        quote_mode=quote_mode,
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
    return GenerationRequest(system_prompt=system_prompt, user_content=user_content, messages=messages)

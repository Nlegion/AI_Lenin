"""DeepSeek-only prompt, R1 excerpts, quote validate, and pipeline isolation tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.analysis.evidence_brief import EvidenceBrief, EvidenceItem
from src.core.generation.deepseek_prompts import (
    DEEPSEEK_STRUCTURE_EXTRA,
    build_deepseek_chat_request,
    build_deepseek_dialectical_chat_request,
)
from src.core.generation.deepseek_quote_validate import (
    deepseek_raw_quote_ok,
    finalize_deepseek_quotes,
    has_no_quote_conflict,
    quote_grounded_in_excerpts,
)
from src.core.generation.deepseek_r1_excerpts import build_deepseek_r1_excerpts
from src.core.generation.pipeline import AnalysisGenerationPipeline
from src.core.generation.prompt_adapter import (
    ANALYSIS_STRUCTURE_EXTRA,
    GIGACHAT_SYSTEM_PROMPT,
    build_chat_request,
    build_dialectical_chat_request,
)
from src.core.generation.quote_allowlist import QuoteCandidate
from src.core.llm.base import GenerationRequest, GenerationResponse
from src.core.safety.news_guard import OutputGuardResult
from src.core.settings.generation_config import load_generation_config
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig


def _r1_item(
    text: str,
    *,
    chunk_id: str = "c1",
    query_used: str = "q",
    score: float = 0.9,
) -> EvidenceItem:
    return EvidenceItem(
        stance_type="core_self",
        source_id="pss-1",
        source_path="data/pss/vol1.txt",
        chunk_id=chunk_id,
        text=text,
        score=score,
        retriever="test",
        query_used=query_used,
    )


def _brief(
    *, r1: list[EvidenceItem], r2: list[EvidenceItem] | None = None
) -> EvidenceBrief:
    return EvidenceBrief(
        news_title="t",
        news_content="c",
        axes=[],
        key_concepts=[],
        r1_core_self=r1,
        r2_influence_agree=r2 or [],
        r3_influence_critical=[],
    )


def test_llama_prompt_still_has_applicability_extra():
    assert "условие применимости" in ANALYSIS_STRUCTURE_EXTRA.lower()
    req = build_chat_request(
        news_title="Инфляция",
        news_content="Рост цен",
        context="[source: x] капитал и труд",
        max_context_chars=500,
    )
    assert "условие применимости" in req.system_prompt.lower()
    assert req.system_prompt.startswith(GIGACHAT_SYSTEM_PROMPT[:40])


def test_llama_dialectical_builder_unchanged_shape():
    req = build_dialectical_chat_request(
        news_title="Инфляция",
        news_content="Рост цен",
        context='## R1\n"капитал"',
        max_context_chars=500,
        quote_mode="principles",
    )
    assert "Доказательная база" in req.user_content
    assert "условие применимости" in req.system_prompt.lower()


def test_deepseek_prompt_omits_applicability_and_requires_quote_when_usable():
    req = build_deepseek_chat_request(
        news_title="Инфляция",
        news_content="Рост цен",
        context="## R1\nтекст",
        max_context_chars=500,
        excerpts_block="Допустимые цитаты:\n- «капитал есть отношение» (pss-1)",
        usable_excerpts=True,
    )
    assert "ограничение/условие применимости" not in req.system_prompt.lower()
    assert ANALYSIS_STRUCTURE_EXTRA not in req.system_prompt
    assert "минимум одну" in req.system_prompt
    assert "Допустимые цитаты" in req.user_content
    assert DEEPSEEK_STRUCTURE_EXTRA.strip()[:20] in req.system_prompt


def test_deepseek_principles_when_no_excerpts():
    req = build_deepseek_dialectical_chat_request(
        news_title="Инфляция",
        news_content="Рост цен",
        context="## R1\n(пусто)",
        max_context_chars=500,
        usable_excerpts=False,
    )
    assert "нет пригодных цитат" in req.system_prompt.lower()
    assert "ограничение/условие применимости" not in req.system_prompt.lower()
    assert ANALYSIS_STRUCTURE_EXTRA not in req.system_prompt


def test_r1_excerpts_ignore_r2_and_apply_thresholds():
    cfg = QualityPostcheckConfig(min_quote_chars=25, min_quote_content_tokens=5)
    long_r1 = (
        "Монополии срастаются с государственным аппаратом и перекладывают "
        "издержки кризиса на трудящиеся массы."
    )
    short = "коротко"
    r2 = EvidenceItem(
        stance_type="influence_agree",
        source_id="r2-src",
        source_path="r2.txt",
        chunk_id="r2",
        text="Совсем другой текст критики империализма для R2 слота проверки.",
        score=0.5,
        retriever="test",
        query_used="q",
    )
    brief = _brief(r1=[_r1_item(long_r1), _r1_item(short, chunk_id="c2")], r2=[r2])
    # Without news filter (empty), keep length-threshold behavior.
    excerpts = build_deepseek_r1_excerpts(brief=brief, config=cfg, news_text="")
    assert excerpts.usable is True
    assert len(excerpts.candidates) == 1
    assert excerpts.candidates[0].chunk_id == "c1"
    assert "Монополии" in excerpts.candidates[0].text
    assert "R2" not in excerpts.block
    assert "Другой текст" not in excerpts.block


def test_r1_excerpts_rank_by_news_and_drop_unrelated():
    cfg = QualityPostcheckConfig(min_quote_chars=25, min_quote_content_tokens=5)
    related = (
        "Инфляция и рост цен на хлеб разоряют трудящиеся массы, "
        "когда монополии вздувают тарифы."
    )
    unrelated = (
        "Философия Гегеля и абстрактная диалектика понятий "
        "требуют особого логического разбора категорий."
    )
    brief = _brief(
        r1=[
            _r1_item(unrelated, chunk_id="u1"),
            _r1_item(related, chunk_id="r1"),
        ]
    )
    news = "Рост цен и инфляция ускорились; тарифы на хлеб выросли."
    excerpts = build_deepseek_r1_excerpts(brief=brief, config=cfg, news_text=news)
    assert excerpts.usable is True
    assert excerpts.candidates[0].chunk_id == "r1"
    assert all(c.chunk_id != "u1" for c in excerpts.candidates)


def test_r1_excerpts_principles_when_no_news_overlap():
    cfg = QualityPostcheckConfig(min_quote_chars=25, min_quote_content_tokens=5)
    unrelated = (
        "Категории гегелевской логики и абстрактные определения "
        "бытия требуют отдельного философского трактата."
    )
    brief = _brief(r1=[_r1_item(unrelated, chunk_id="u1", score=0.1)])
    excerpts = build_deepseek_r1_excerpts(
        brief=brief,
        config=cfg,
        news_text="Футбольный матч закончился ничьей без голов.",
    )
    # Soft fallback keeps top-k so live RAG can still quote; ranking stays low.
    assert excerpts.usable is True
    assert excerpts.candidates[0].chunk_id == "u1"
    assert float(excerpts.candidates[0].meta.get("link_score") or 0) < 0.12
    assert "Связь с новостью слабая" in excerpts.block


def test_r1_excerpts_theme_bridge_and_query_boost():
    cfg = QualityPostcheckConfig(min_quote_chars=25, min_quote_content_tokens=5)
    political = (
        "Буржуазное правительство скатывается к реформизму и теряет опору в массах "
        "пролетариата при парламентских кризисах."
    )
    abstract = (
        "Категории гегелевской логики и абстрактные определения "
        "бытия требуют отдельного философского трактата."
    )
    brief = _brief(
        r1=[
            _r1_item(abstract, chunk_id="a1", query_used="философия", score=0.2),
            _r1_item(
                political,
                chunk_id="p1",
                query_used="рейтинг канцлера парламент кризис",
                score=0.4,
            ),
        ]
    )
    news = "Опрос показал рекордное падение рейтинга канцлера в парламенте."
    excerpts = build_deepseek_r1_excerpts(brief=brief, config=cfg, news_text=news)
    assert excerpts.candidates[0].chunk_id == "p1"
    assert float(excerpts.candidates[0].meta.get("theme_bonus") or 0) > 0
    assert float(excerpts.best_link_score) >= 0.12


def test_scrub_removes_attribution_and_unclosed_quote():
    broken = (
        "Факт: Цены выросли по данным биржи.\n\n"
        "Механизм: Как писал Ленин, «америка окажет Здесь мы видим тот же\n\n"
        "Механизм: Страны идут туда где видят экономический интерес капитала.\n\n"
        "Вывод: Это прагматичный расчет монополий а не дружба народов."
    )
    cleaned, flags = finalize_deepseek_quotes(
        text=broken,
        excerpts=[],
        usable_excerpts=False,
    )
    assert "как писал" not in cleaned.lower()
    assert "«" not in cleaned
    assert "америка" not in cleaned.lower()
    assert "страны идут" in cleaned.lower()
    assert flags["deepseek_stripped_all_quotes"] or flags["deepseek_scrubbed_debris"]


def test_repair_colon_holes_and_fragment_sentences():
    broken = (
        "Факт: Умер филолог.\n\n"
        "Механизм: В контексте прямо указан тип такого деятеля :. "
        "Это пример. и восстают против этого другие пытаются включить "
        "империализм в ход развития.\n\n"
        "Вывод: Наука должна служить народу, а не классу эксплуататоров."
    )
    cleaned, flags = finalize_deepseek_quotes(
        text=broken,
        excerpts=[],
        usable_excerpts=False,
    )
    assert ":." not in cleaned
    assert "деятеля :" not in cleaned
    assert not cleaned.lower().strip().startswith("и восстают")
    assert "наука должна служить" in cleaned.lower()
    assert flags.get("deepseek_repaired_holes") is True or ":." not in cleaned


def test_repair_drops_stump_and_glue_fragments():
    broken = (
        "Факт: В лесу найдено тело адвоката.\n\n"
        "Механизм: Суд не устанавливает истину. "
        "Именно поэтому «лакеи судьи» истории — принцип, описанный Равнодушие "
        "и формализм чиновников от юстиции. "
        "Вопрос о т. Этот принцип напрямую объясняет суть.\n\n"
        "Вывод: Смерть адвоката — закономерное проявление гнилостности буржуазной юстиции."
    )
    cleaned, _flags = finalize_deepseek_quotes(
        text=broken,
        excerpts=[],
        usable_excerpts=False,
    )
    assert "о т." not in cleaned.lower()
    assert "описанный равнодушие" not in cleaned.lower()
    assert "гнилостности буржуазной юстиции" in cleaned.lower()
    text = (
        "Механизм: Показная кулинарная не решает главного. "
        "ясо народу не по карману », и любые зрелищные мероприятия остаются "
        "лишь вывеской классового разрыва между богатыми и беднотой."
    )
    cleaned, _flags = finalize_deepseek_quotes(
        text=text,
        excerpts=[],
        usable_excerpts=False,
    )
    assert "»" not in cleaned
    assert "по карману" not in cleaned.lower()
    assert "классового разрыва" in cleaned.lower() or "мероприятия" in cleaned.lower()


def test_scrub_keeps_attribution_before_grounded_quote():
    excerpts = [
        QuoteCandidate(
            text="Монополии срастаются с государственным аппаратом",
            chunk_id="c1",
            source_id="pss-1",
        )
    ]
    text = (
        "Механизм: Как писал Ленин, «Монополии срастаются с государственным аппаратом» "
        "в условиях кризиса."
    )
    cleaned, _flags = finalize_deepseek_quotes(
        text=text,
        excerpts=excerpts,
        usable_excerpts=True,
    )
    assert "Монополии срастаются" in cleaned
    assert "«" in cleaned
    # Attribution before a real quote is allowed.
    assert "писал" in cleaned.lower() or "монополии" in cleaned.lower()


def test_quote_validate_accepts_grounded_and_rejects_news_or_r2():
    excerpts = [
        QuoteCandidate(
            text="Монополии срастаются с государственным аппаратом",
            chunk_id="c1",
            source_id="pss-1",
        )
    ]
    ok = (
        "Факт: x\n"
        "Механизм: Как писал Ленин, «Монополии срастаются с государственным аппаратом».\n"
        "Вывод: y"
    )
    assert quote_grounded_in_excerpts(text=ok, excerpts=excerpts) is True
    bad = "Механизм: «Совсем другой текст критики империализма»"
    assert quote_grounded_in_excerpts(text=bad, excerpts=excerpts) is False
    assert quote_grounded_in_excerpts(text="без кавычек", excerpts=excerpts) is False


def test_no_quote_conflict_and_finalize_strips_quotes():
    conflict = (
        "Факт: x\n"
        "Механизм: В предоставленном контексте подходящей цитаты нет. "
        "И всё же «обломок фразы».\n"
        "Вывод: y"
    )
    assert has_no_quote_conflict(conflict) is True
    assert (
        deepseek_raw_quote_ok(
            text=conflict,
            excerpts=[],
            usable_excerpts=True,
        )
        is False
    )
    cleaned, flags = finalize_deepseek_quotes(
        text=conflict,
        excerpts=[],
        usable_excerpts=False,
    )
    assert "«" not in cleaned
    assert flags["deepseek_stripped_all_quotes"] is True
    assert "подходящей цитаты нет" in cleaned.lower()


def test_finalize_keeps_grounded_drops_ungrounded():
    excerpts = [
        QuoteCandidate(
            text="Монополии срастаются с государственным аппаратом",
            chunk_id="c1",
            source_id="pss-1",
        )
    ]
    text = (
        "Механизм: «Монополии срастаются с государственным аппаратом» "
        "и ещё «выдуманный обрывок»."
    )
    cleaned, flags = finalize_deepseek_quotes(
        text=text,
        excerpts=excerpts,
        usable_excerpts=True,
    )
    assert "Монополии срастаются" in cleaned
    assert "выдуманный" not in cleaned
    assert flags["deepseek_stripped_ungrounded"] is True


class _CountingBackend:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls = 0
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        self.requests.append(request)
        text = self.texts[min(self.calls - 1, len(self.texts) - 1)]
        return GenerationResponse(
            text=text,
            backend="mock",
            model_name="deepseek-v4-flash",
            latency_ms=10,
            finish_reason="stop",
        )

    async def close(self) -> None:
        return None


def _pipeline_with_backend(
    backend, *, provider: str = "llama"
) -> AnalysisGenerationPipeline:
    import os

    # Isolate from ambient LLM_* env left by live DeepSeek runs in the same shell.
    ambient_keys = (
        "LLM_SPAWN_LOCAL",
        "LLM_PROVIDER",
        "LLM_MODEL_NAME",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "GENERATION_SERVER_URL",
        "LLM_DEEPSEEK_ALLOW_INSECURE_URL",
        "LLM_THINKING_MODE",
        "LLM_REASONING_EFFORT",
    )
    saved = {key: os.environ.pop(key) for key in ambient_keys if key in os.environ}
    try:
        config = load_generation_config(Path("config/generation.yaml"))
        payload = config.model_dump()
        payload["provider"] = provider
        payload["spawn_local"] = provider != "deepseek"
        if provider == "deepseek":
            payload["server_url"] = "https://api.deepseek.com"
            payload["api_key"] = "sk-test"
        from src.core.settings.generation_config import GenerationConfig

        config = GenerationConfig.model_validate(payload)
    finally:
        os.environ.update(saved)
    guard = MagicMock()
    guard.mark_unverified_facts.side_effect = lambda analysis, **_k: (analysis, [])
    guard.guard_output.side_effect = lambda analysis, **_k: OutputGuardResult(
        blocked=False,
        moderated_text=analysis,
        reason_codes=[],
    )
    pipeline = AnalysisGenerationPipeline(
        base_dir=Path("."),
        context_builder=lambda _query: "context",
        news_guard=guard,
        generation_config=config,
        persona_model="base_strong",
    )
    pipeline.backend = backend
    pipeline.dialectical_enabled = False
    return pipeline


@pytest.mark.asyncio
async def test_llama_pipeline_one_generate_call():
    backend = _CountingBackend(
        [
            "Факт: цены выросли.\n"
            "Механизм: капиталистическая конкуренция.\n"
            "Вывод: трудящиеся платят."
        ]
    )
    pipeline = _pipeline_with_backend(backend, provider="llama")
    result = await pipeline._generate_with_context(
        news_title="Рост цен",
        news_content="Инфляция ускорилась в регионе",
        context="[source: x] капитал",
        feedback=None,
        warn_only_guard=True,
        brief=None,
        orchestration_mode="legacy",
        dialectical_prompt=False,
    )
    assert backend.calls == 1
    assert "Факт" in result.analysis
    await pipeline.close()


@pytest.mark.asyncio
async def test_deepseek_valid_quote_single_generate():
    quote = (
        "Монополии срастаются с государственным аппаратом и перекладывают "
        "издержки кризиса на трудящиеся массы."
    )
    brief = _brief(r1=[_r1_item(quote)])
    backend = _CountingBackend(
        [
            "Факт: цены выросли.\n"
            f"Механизм: «{quote}» — основа давления на массы.\n"
            "Вывод: кризис перекладывают на трудящихся."
        ]
    )
    pipeline = _pipeline_with_backend(backend, provider="deepseek")
    result = await pipeline._generate_with_context(
        news_title="Кризис монополий",
        news_content="Монополии перекладывают издержки кризиса на трудящиеся массы",
        context=brief.render_for_prompt(),
        feedback=None,
        warn_only_guard=True,
        brief=brief,
        orchestration_mode="dialectical_v1",
        dialectical_prompt=True,
    )
    assert backend.calls == 1
    assert result.metadata.get("deepseek_quote_valid") is True
    assert result.metadata.get("deepseek_regen_count") == 0
    assert result.metadata.get("deepseek_quote_unfulfilled") is False
    assert "«" in result.analysis or '"' in result.analysis
    await pipeline.close()


@pytest.mark.asyncio
async def test_deepseek_missing_quote_triggers_one_regen():
    quote = (
        "Монополии срастаются с государственным аппаратом и перекладывают "
        "издержки кризиса на трудящиеся массы."
    )
    brief = _brief(r1=[_r1_item(quote)])
    backend = _CountingBackend(
        [
            "Факт: цены выросли.\nМеханизм: без опоры на текст.\nВывод: плохо.",
            "Факт: цены выросли.\n"
            f"Механизм: «{quote}» объясняет перенос издержек.\n"
            "Вывод: массы платят.",
        ]
    )
    pipeline = _pipeline_with_backend(backend, provider="deepseek")
    result = await pipeline._generate_with_context(
        news_title="Кризис монополий",
        news_content="Монополии перекладывают издержки кризиса на трудящиеся массы",
        context=brief.render_for_prompt(),
        feedback=None,
        warn_only_guard=True,
        brief=brief,
        orchestration_mode="dialectical_v1",
        dialectical_prompt=True,
    )
    assert backend.calls == 2
    assert result.metadata.get("deepseek_regen_count") == 1
    assert result.latency_ms == 20
    assert result.metadata.get("deepseek_quote_valid") is True
    await pipeline.close()


@pytest.mark.asyncio
async def test_deepseek_empty_r1_principles_one_call():
    brief = _brief(r1=[])
    backend = _CountingBackend(
        [
            "Факт: цены выросли.\n"
            "Механизм: принципы без цитат.\n"
            "Вывод: массы несут издержки."
        ]
    )
    pipeline = _pipeline_with_backend(backend, provider="deepseek")
    result = await pipeline._generate_with_context(
        news_title="Рост цен",
        news_content="Инфляция ускорилась в регионе",
        context="## R1\n(пусто)",
        feedback=None,
        warn_only_guard=True,
        brief=brief,
        orchestration_mode="dialectical_v1",
        dialectical_prompt=True,
    )
    assert backend.calls == 1
    assert "нет пригодных цитат" in backend.requests[0].system_prompt.lower()
    assert result.metadata.get("deepseek_regen_count") == 0
    assert result.metadata.get("deepseek_quote_unfulfilled") is False
    await pipeline.close()


@pytest.mark.asyncio
async def test_deepseek_second_fail_sets_unfulfilled():
    quote = (
        "Монополии срастаются с государственным аппаратом и перекладывают "
        "издержки кризиса на трудящиеся массы."
    )
    brief = _brief(r1=[_r1_item(quote)])
    fabricated = (
        "Факт: цены выросли.\n"
        "Механизм: «выдуманная цитата которой нет в корпусе совсем».\n"
        "Вывод: итог."
    )
    backend = _CountingBackend([fabricated, fabricated])
    pipeline = _pipeline_with_backend(backend, provider="deepseek")
    result = await pipeline._generate_with_context(
        news_title="Кризис монополий",
        news_content="Монополии перекладывают издержки кризиса на трудящиеся массы",
        context=brief.render_for_prompt(),
        feedback=None,
        warn_only_guard=True,
        brief=brief,
        orchestration_mode="dialectical_v1",
        dialectical_prompt=True,
    )
    assert backend.calls == 2
    assert result.metadata.get("deepseek_quote_unfulfilled") is True
    assert result.metadata.get("deepseek_quote_valid") is False
    # Fabricated quote should be stripped by postcheck and/or DeepSeek cleanup.
    assert "выдуманная цитата" not in result.analysis
    await pipeline.close()


@pytest.mark.asyncio
async def test_deepseek_conflict_phrase_plus_quotes_regens_or_strips():
    quote = (
        "Монополии срастаются с государственным аппаратом и перекладывают "
        "издержки кризиса на трудящиеся массы."
    )
    brief = _brief(r1=[_r1_item(quote)])
    conflict = (
        "Факт: цены выросли.\n"
        "Механизм: В предоставленном контексте подходящей цитаты нет. "
        "Но всё же «обломок».\n"
        "Вывод: массы платят."
    )
    fixed = (
        "Факт: цены выросли.\n"
        f"Механизм: «{quote}» объясняет перенос издержек.\n"
        "Вывод: массы платят."
    )
    backend = _CountingBackend([conflict, fixed])
    pipeline = _pipeline_with_backend(backend, provider="deepseek")
    result = await pipeline._generate_with_context(
        news_title="Кризис монополий",
        news_content="Монополии перекладывают издержки кризиса на трудящиеся массы",
        context=brief.render_for_prompt(),
        feedback=None,
        warn_only_guard=True,
        brief=brief,
        orchestration_mode="dialectical_v1",
        dialectical_prompt=True,
    )
    assert backend.calls == 2
    assert result.metadata.get("deepseek_regen_count") == 1
    assert result.metadata.get("deepseek_quote_valid") is True
    assert "обломок" not in result.analysis
    await pipeline.close()

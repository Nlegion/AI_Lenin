from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.generation.chat_backend import ChatCompletionsBackend
from src.core.generation.factory import build_generation_backend
from src.core.generation.fallback import recommend_persona_model
from src.core.generation.pipeline import AnalysisGenerationPipeline
from src.core.generation.prompt_adapter import (
    GIGACHAT_SYSTEM_PROMPT,
    build_chat_request,
)
from src.core.safety.news_guard import OutputGuardResult
from src.core.settings.generation_config import (
    FallbackConfig,
    SafetyConfig,
    load_generation_config,
)


def test_default_persona_is_base_strong():
    config = load_generation_config(Path("config/generation.yaml"))
    assert config.persona_model == "base_strong"
    assert config.active_backend().api_style == "chat_completions"
    assert "GigaChat3" in config.active_backend().model_path


def test_gigachat_system_prompt_contains_hard_bans():
    assert "насилию" in GIGACHAT_SYSTEM_PROMPT
    assert "образовательн" in GIGACHAT_SYSTEM_PROMPT
    assert (
        "политика безопасности" in GIGACHAT_SYSTEM_PROMPT
    )  # instruct NOT to use template
    assert "Анализ данной темы невозможен" not in GIGACHAT_SYSTEM_PROMPT
    request = build_chat_request(
        news_title="Инфляция",
        news_content="Рост цен и безработица",
        context="[source: том 1] капитал",
        max_context_chars=1000,
    )
    assert request.messages[0]["role"] == "system"
    assert request.messages[1]["role"] == "user"
    assert "Контекст RAG" in request.messages[1]["content"]


def test_factory_builds_chat_backend_by_default():
    backend, cfg = build_generation_backend(
        base_dir=Path("."), persona_model="base_strong"
    )
    assert isinstance(backend, ChatCompletionsBackend)
    assert cfg.persona_model == "base_strong"


def test_fallback_recommendation_stays_base_strong_when_enabled(tmp_path: Path):
    config = load_generation_config(Path("config/generation.yaml"))
    audit = tmp_path / "audit.jsonl"
    audit.write_text("\n".join(['{"high_risk": true}'] * 6) + "\n", encoding="utf-8")
    config.safety = SafetyConfig(
        post_filter=True,
        fallback=FallbackConfig(
            enabled=True,
            incident_threshold=5,
            window_events=50,
            audit_log_path=str(audit),
        ),
    )
    assert recommend_persona_model(config=config, base_dir=Path(".")) == "base_strong"


def test_cli_persona_model_rejects_fine_tuned():
    from scripts.quality.run_quality_qa_batch_cli import build_parser as quality_parser
    from scripts.quality.run_live_news_qa_batch_cli import build_parser as live_parser
    from scripts.quality.run_live_news_qa_24h_cli import build_parser as live24_parser

    for build in (quality_parser, live_parser, live24_parser):
        parser = build()
        with pytest.raises(SystemExit):
            parser.parse_args(["--persona-model", "fine_tuned"])


def test_fallback_disabled_keeps_active_model(tmp_path: Path):
    config = load_generation_config(Path("config/generation.yaml"))
    audit = tmp_path / "audit.jsonl"
    audit.write_text("\n".join(['{"high_risk": true}'] * 6) + "\n", encoding="utf-8")
    config.safety.fallback.enabled = False
    config.safety.fallback.audit_log_path = str(audit)
    assert recommend_persona_model(config=config, base_dir=Path(".")) == "base_strong"


@pytest.mark.asyncio
async def test_chat_backend_request_shape_omits_tool_choice():
    backend = ChatCompletionsBackend(
        server_url="http://127.0.0.1:8080",
        backend_config=load_generation_config(
            Path("config/generation.yaml")
        ).active_backend(),
        persona_model="base_strong",
    )
    response_json = {
        "choices": [{"message": {"content": "Анализ в образовательной рамке."}}],
    }
    response_ctx = AsyncMock()
    response_ctx.__aenter__.return_value.status = 200
    response_ctx.__aenter__.return_value.json = AsyncMock(return_value=response_json)
    session = MagicMock()
    session.post.return_value = response_ctx
    backend.session = session

    request = build_chat_request(
        news_title="Экономика",
        news_content="Рост цен",
        context="[source: том 1] капитал",
        max_context_chars=500,
    )
    result = await backend.generate(request=request)
    assert result.text
    kwargs = session.post.call_args.kwargs
    payload = kwargs["json"]
    assert session.post.call_args.args[0].endswith("/v1/chat/completions")
    assert "tool_choice" not in payload
    assert payload["messages"][0]["role"] == "system"
    await backend.close()


@pytest.mark.asyncio
async def test_pipeline_applies_newsguard_post_filter():
    config = load_generation_config(Path("config/generation.yaml"))
    guard = MagicMock()
    guard.mark_unverified_facts.return_value = ("raw analysis", [])
    guard.guard_output.return_value = OutputGuardResult(
        blocked=True,
        moderated_text="SAFE_TEMPLATE",
        reason_codes=["blocked_military"],
    )

    class _FakeBackend:
        async def generate(self, request):  # noqa: ANN001
            from src.core.llm.base import GenerationResponse

            return GenerationResponse(
                text="raw analysis",
                backend="base_strong",
                model_name="giga",
                latency_ms=1,
            )

        async def close(self) -> None:
            return None

    pipeline = AnalysisGenerationPipeline(
        base_dir=Path("."),
        context_builder=lambda _query: "context",
        news_guard=guard,
        generation_config=config,
        persona_model="base_strong",
    )
    pipeline.backend = _FakeBackend()
    result = await pipeline.generate(
        news_title="Новость",
        news_content="Текст",
        enhanced_query="query",
    )
    assert result.analysis == "SAFE_TEMPLATE"
    assert result.guard_result.blocked is True
    guard.guard_output.assert_called_once()
    await pipeline.close()

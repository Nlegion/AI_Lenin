"""Remote LLM seam and env override tests (no live network / Docker)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.generation.prompt_adapter import build_chat_request
from src.core.llm.chat_completions import ChatCompletionsBackend
from src.core.llm.deepseek import DeepSeekBackend
from src.core.llm.factory import build_generation_backend
from src.core.processor import NewsProcessor
from src.core.settings.generation_config import (
    apply_generation_env_overrides,
    load_generation_config,
    llm_spawn_local_from_env,
    normalize_server_url,
)


def test_normalize_server_url_strips_trailing_v1():
    assert normalize_server_url("https://api.deepseek.com/v1/") == (
        "https://api.deepseek.com"
    )
    assert normalize_server_url("https://api.deepseek.com") == (
        "https://api.deepseek.com"
    )


def test_llm_spawn_local_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_SPAWN_LOCAL", raising=False)
    assert llm_spawn_local_from_env() is True
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    assert llm_spawn_local_from_env() is False
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "1")
    assert llm_spawn_local_from_env() is True


def test_env_overrides_url_api_key_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("GENERATION_SERVER_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-chat")
    config = load_generation_config(Path("config/generation.yaml"))
    assert config.spawn_local is False
    assert config.provider == "llama"
    assert config.server_url == "https://api.deepseek.com"
    assert config.api_key == "sk-test"
    assert config.active_backend().model_name == "deepseek-chat"


def test_remote_mode_requires_model_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
    from src.core.settings.generation_config import GenerationConfig

    payload = GenerationConfig.model_validate(
        {
            "persona_model": "base_strong",
            "server_url": "http://127.0.0.1:8080",
            "backends": {
                "base_strong": {
                    "model_name": "local",
                    "model_path": "models/x.gguf",
                }
            },
        }
    )
    with pytest.raises(ValueError, match="LLM_MODEL_NAME"):
        apply_generation_env_overrides(payload)


def test_processor_skips_lenin_server_when_remote(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")

    def _close_created_coroutine(coroutine):
        coroutine.close()
        return None

    with patch(
        "src.core.processor.asyncio.create_task",
        side_effect=_close_created_coroutine,
    ):
        with patch("src.core.processor.TelegramPublisher"):
            processor = NewsProcessor()
    assert processor.spawn_local is False
    assert processor.server is None


@pytest.mark.asyncio
async def test_processor_initialize_skips_start_server(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-chat")

    def _close_created_coroutine(coroutine):
        coroutine.close()
        return None

    with patch(
        "src.core.processor.asyncio.create_task",
        side_effect=_close_created_coroutine,
    ):
        with patch("src.core.processor.TelegramPublisher") as pub_cls:
            publisher = MagicMock()
            publisher.send_admin_notification = AsyncMock()
            pub_cls.return_value = publisher
            processor = NewsProcessor()
            processor.publisher = publisher

    analyzer = MagicMock()
    analyzer.initialize_session = AsyncMock()
    with patch("src.core.processor.LeninAnalyzer", return_value=analyzer):
        with patch("src.core.processor.LeninServer") as server_cls:
            await processor.initialize_components()
            server_cls.assert_not_called()
    assert processor.analyzer is analyzer
    assert processor.analyzer_ready.is_set()


@pytest.mark.asyncio
async def test_remote_payload_omits_llama_fields_and_sends_bearer():
    backend = ChatCompletionsBackend(
        server_url="https://api.deepseek.com",
        backend_config=load_generation_config(
            Path("config/generation.yaml")
        ).active_backend(),
        persona_model="base_strong",
        api_key="sk-secret",
        spawn_local=False,
    )
    response_json = {
        "choices": [{"message": {"content": "ok"}}],
    }
    response_ctx = AsyncMock()
    response_ctx.__aenter__.return_value.status = 200
    response_ctx.__aenter__.return_value.json = AsyncMock(return_value=response_json)
    session = MagicMock()
    session.post.return_value = response_ctx
    backend.session = session

    request = build_chat_request(
        news_title="t",
        news_content="c",
        context="ctx",
        max_context_chars=100,
    )
    await backend.generate(request=request)
    payload = session.post.call_args.kwargs["json"]
    headers = session.post.call_args.kwargs["headers"]
    assert "repetition_penalty" not in payload
    assert "seed" not in payload
    assert headers["Authorization"] == "Bearer sk-secret"
    assert session.post.call_args.args[0] == (
        "https://api.deepseek.com/v1/chat/completions"
    )
    await backend.close()


def test_factory_passes_remote_fields(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("GENERATION_SERVER_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL_NAME", "remote-model")
    backend, cfg = build_generation_backend(base_dir=Path("."))
    assert isinstance(backend, ChatCompletionsBackend)
    assert not isinstance(backend, DeepSeekBackend)
    assert backend.spawn_local is False
    assert backend.api_key == "key"
    assert backend.server_url == "https://api.example.com"
    assert cfg.active_backend().model_name == "remote-model"
    assert cfg.provider == "llama"

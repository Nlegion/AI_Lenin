"""DeepSeek provider adapter and config validation tests (no live network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.generation.prompt_adapter import build_chat_request
from src.core.llm.chat_completions import ChatCompletionsBackend
from src.core.llm.deepseek import DeepSeekBackend
from src.core.llm.factory import build_generation_backend
from src.core.settings.deepseek_config import (
    DEEPSEEK_DEFAULT_MODEL,
    DEEPSEEK_DEFAULT_SERVER_URL,
)
from src.core.settings.generation_config import (
    GenerationConfig,
    apply_generation_env_overrides,
    load_generation_config,
)


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_SPAWN_LOCAL",
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "LLM_MODEL_NAME",
        "GENERATION_SERVER_URL",
        "LLM_DEEPSEEK_ALLOW_INSECURE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_provider_remains_llama(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    config = load_generation_config(Path("config/generation.yaml"))
    assert config.provider == "llama"
    assert config.spawn_local is True
    backend, cfg = build_generation_backend(base_dir=Path("."), config=config)
    assert isinstance(backend, ChatCompletionsBackend)
    assert not isinstance(backend, DeepSeekBackend)
    assert cfg.provider == "llama"


def test_deepseek_provider_defaults_url_and_model(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    config = load_generation_config(Path("config/generation.yaml"))
    assert config.provider == "deepseek"
    assert config.server_url == DEEPSEEK_DEFAULT_SERVER_URL
    assert config.api_key == "sk-deepseek"
    assert config.active_backend().model_name == DEEPSEEK_DEFAULT_MODEL


def test_llm_api_key_wins_over_deepseek_key(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_API_KEY", "sk-primary")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secondary")
    config = load_generation_config(Path("config/generation.yaml"))
    assert config.api_key == "sk-primary"


def test_deepseek_key_ignored_for_llama_provider(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "llama")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("LLM_MODEL_NAME", "remote-model")
    monkeypatch.setenv("GENERATION_SERVER_URL", "https://api.example.com")
    config = load_generation_config(Path("config/generation.yaml"))
    assert config.provider == "llama"
    assert config.api_key is None


def test_deepseek_rejects_local_spawn(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "true")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    with pytest.raises(ValueError, match="LLM_SPAWN_LOCAL=false"):
        load_generation_config(Path("config/generation.yaml"))


def test_deepseek_rejects_missing_key(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_generation_config(Path("config/generation.yaml"))


def test_invalid_provider_raises(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
        load_generation_config(Path("config/generation.yaml"))


def test_deepseek_rejects_insecure_url(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("GENERATION_SERVER_URL", "http://127.0.0.1:9000")
    with pytest.raises(ValueError, match="HTTPS remote URL"):
        load_generation_config(Path("config/generation.yaml"))


def test_deepseek_allows_insecure_url_with_escape(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("GENERATION_SERVER_URL", "http://proxy.local:8080")
    monkeypatch.setenv("LLM_DEEPSEEK_ALLOW_INSECURE_URL", "true")
    config = load_generation_config(Path("config/generation.yaml"))
    assert config.server_url == "http://proxy.local:8080"


def test_factory_selects_deepseek_backend(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    backend, cfg = build_generation_backend(base_dir=Path("."))
    assert isinstance(backend, DeepSeekBackend)
    assert cfg.provider == "deepseek"


def test_generic_remote_without_provider_stays_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("GENERATION_SERVER_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL_NAME", "arbitrary-remote-model")
    backend, cfg = build_generation_backend(base_dir=Path("."))
    assert isinstance(backend, ChatCompletionsBackend)
    assert not isinstance(backend, DeepSeekBackend)
    assert cfg.provider == "llama"
    assert cfg.active_backend().model_name == "arbitrary-remote-model"


@pytest.mark.asyncio
async def test_deepseek_payload_and_endpoint(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    config = load_generation_config(Path("config/generation.yaml"))
    backend = DeepSeekBackend(
        server_url=config.server_url,
        backend_config=config.active_backend(),
        persona_model="base_strong",
        api_key=config.api_key,
        spawn_local=False,
    )
    response_json = {"choices": [{"message": {"content": "ok"}}]}
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
    assert session.post.call_args.args[0] == (
        f"{DEEPSEEK_DEFAULT_SERVER_URL}/chat/completions"
    )
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
    assert "repetition_penalty" not in payload
    assert "seed" not in payload
    await backend.close()


@pytest.mark.asyncio
async def test_deepseek_thinking_enabled_includes_effort():
    backend_cfg = GenerationConfig.model_validate(
        {
            "persona_model": "base_strong",
            "provider": "deepseek",
            "backends": {
                "base_strong": {
                    "model_name": "deepseek-v4-flash",
                    "model_path": "models/x.gguf",
                    "thinking_mode": "enabled",
                    "reasoning_effort": "low",
                }
            },
        }
    ).active_backend()
    backend = DeepSeekBackend(
        server_url=DEEPSEEK_DEFAULT_SERVER_URL,
        backend_config=backend_cfg,
        spawn_local=False,
        api_key="sk",
    )
    response_json = {"choices": [{"message": {"content": "ok"}}]}
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
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "low"
    assert "temperature" in payload
    assert "top_p" in payload
    await backend.close()


@pytest.mark.asyncio
async def test_local_default_payload_keeps_llama_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_llm_env(monkeypatch)
    config = load_generation_config(Path("config/generation.yaml"))
    backend = ChatCompletionsBackend(
        server_url=config.server_url,
        backend_config=config.active_backend(),
        spawn_local=True,
    )
    response_json = {"choices": [{"message": {"content": "ok"}}]}
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
    assert session.post.call_args.args[0].endswith("/v1/chat/completions")
    assert "repetition_penalty" in payload
    assert "seed" in payload
    await backend.close()


def test_apply_overrides_centralized_for_deepseek(monkeypatch: pytest.MonkeyPatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_SPAWN_LOCAL", "false")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    base = GenerationConfig.model_validate(
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
    config = apply_generation_env_overrides(base)
    assert config.provider == "deepseek"
    assert config.server_url == DEEPSEEK_DEFAULT_SERVER_URL

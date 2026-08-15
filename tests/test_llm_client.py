"""LLM client package tests (mocked HTTP; no live llama-server)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.generation.prompt_adapter import build_chat_request
from src.core.llm.chat_completions import ChatCompletionsBackend
from src.core.llm.factory import build_generation_backend
from src.core.llm.health import is_llama_server_active
from src.core.settings.generation_config import load_generation_config


def test_factory_builds_chat_backend_from_llm_package():
    backend, cfg = build_generation_backend(
        base_dir=Path("."), persona_model="base_strong"
    )
    assert isinstance(backend, ChatCompletionsBackend)
    assert cfg.persona_model == "base_strong"


def test_factory_rejects_unsupported_api_style():
    real = load_generation_config(Path("config/generation.yaml"))
    backend_cfg = real.active_backend().model_copy(deep=True)
    object.__setattr__(backend_cfg, "api_style", "completion")

    class _Cfg:
        persona_model = "base_strong"
        server_url = real.server_url

        def with_persona_model(self, persona_model):  # noqa: ANN001
            self.persona_model = persona_model
            return self

        def active_backend(self):
            return backend_cfg

    with pytest.raises(ValueError, match="Unsupported api_style"):
        build_generation_backend(
            base_dir=Path("."),
            config=_Cfg(),  # type: ignore[arg-type]
            persona_model="base_strong",
        )


@pytest.mark.asyncio
async def test_chat_completions_request_shape():
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
    assert session.post.call_args.args[0].endswith("/v1/chat/completions")
    assert "tool_choice" not in session.post.call_args.kwargs["json"]
    await backend.close()


def test_shim_imports_remain_callable():
    from src.core.generation.factory import build_generation_backend as shim_factory
    from src.core.llama_server import LeninServer
    from src.core.settings.device import is_llama_server_active as device_probe
    from src.core.settings.llama_runtime import LlamaRuntimePaths, resolve_llama_runtime
    from src.core.llm.factory import build_generation_backend as llm_factory
    from src.core.llm.runtime import resolve_llama_runtime as llm_resolve

    assert shim_factory is llm_factory
    assert resolve_llama_runtime is llm_resolve
    assert LeninServer is not None
    assert callable(device_probe)
    assert LlamaRuntimePaths is not None


def test_health_probe_uses_fake_http():
    with patch("src.core.llm.health.urlopen") as mock_open:
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        mock_open.return_value = response
        assert is_llama_server_active(
            server_url="http://127.0.0.1:8080", timeout_sec=0.1
        )
        mock_open.assert_called()


def test_health_probe_rejects_non_http_scheme():
    assert not is_llama_server_active(server_url="file:///tmp/x", timeout_sec=0.1)

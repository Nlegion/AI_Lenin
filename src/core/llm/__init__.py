"""LLM client package: Protocol, HTTP backend, factory, process lifecycle.

Lazy exports avoid eager imports that would couple retrieval/settings consumers
to the full LLM stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ChatCompletionsBackend",
    "DeepSeekBackend",
    "GenerationBackend",
    "GenerationRequest",
    "GenerationResponse",
    "LeninServer",
    "LlamaRuntimePaths",
    "build_generation_backend",
    "is_llama_server_active",
    "resolve_llama_runtime",
]


def __getattr__(name: str) -> Any:
    if name in {"GenerationRequest", "GenerationResponse", "GenerationBackend"}:
        from src.core.llm import base as _base

        return getattr(_base, name)
    if name == "ChatCompletionsBackend":
        from src.core.llm.chat_completions import ChatCompletionsBackend

        return ChatCompletionsBackend
    if name == "DeepSeekBackend":
        from src.core.llm.deepseek import DeepSeekBackend

        return DeepSeekBackend
    if name == "build_generation_backend":
        from src.core.llm.factory import build_generation_backend

        return build_generation_backend
    if name == "LeninServer":
        from src.core.llm.server import LeninServer

        return LeninServer
    if name == "is_llama_server_active":
        from src.core.llm.health import is_llama_server_active

        return is_llama_server_active
    if name in {"LlamaRuntimePaths", "resolve_llama_runtime"}:
        from src.core.llm import runtime as _runtime

        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")

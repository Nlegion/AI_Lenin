"""Compatibility shim. Prefer: from src.core.llm.chat_completions import ChatCompletionsBackend"""

from src.core.llm.chat_completions import ChatCompletionsBackend

__all__ = ["ChatCompletionsBackend"]

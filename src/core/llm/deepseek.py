"""DeepSeek chat-completions adapter (OpenAI-compatible remote API)."""

from __future__ import annotations

from src.core.llm.base import GenerationRequest
from src.core.llm.chat_completions import ChatCompletionsBackend


class DeepSeekBackend(ChatCompletionsBackend):
    """Thin DeepSeek adapter over shared chat-completions transport."""

    def _completions_path(self) -> str:
        return "/chat/completions"

    def _build_payload(self, request: GenerationRequest) -> dict:
        thinking_mode = getattr(self.backend_config, "thinking_mode", "disabled")
        payload: dict = {
            "model": self.backend_config.model_name,
            "messages": request.messages,
            "temperature": self.backend_config.temperature,
            "top_p": self.backend_config.top_p,
            "max_tokens": self.backend_config.max_tokens,
            "stream": False,
            "thinking": {"type": thinking_mode},
        }
        if thinking_mode == "enabled":
            effort = getattr(self.backend_config, "reasoning_effort", None)
            if effort:
                payload["reasoning_effort"] = effort
        return payload

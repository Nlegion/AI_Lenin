"""GigaChat3 OpenAI-compatible chat completions backend."""

from __future__ import annotations

from time import perf_counter

import aiohttp

from src.core.generation.base import GenerationRequest, GenerationResponse
from src.core.settings.generation_config import BackendConfig


class ChatCompletionsBackend:
    def __init__(
        self,
        *,
        server_url: str,
        backend_config: BackendConfig,
        session: aiohttp.ClientSession | None = None,
        persona_model: str = "base_strong",
    ):
        self.server_url = server_url.rstrip("/")
        self.backend_config = backend_config
        self.session = session
        self.persona_model = persona_model
        self._owns_session = False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=300, sock_connect=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self.session

    async def close(self) -> None:
        if self._owns_session and self.session is not None:
            await self.session.close()
            self.session = None
            self._owns_session = False

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        session = await self._ensure_session()
        payload = {
            "model": self.backend_config.model_name,
            "tool_choice": "none",
            "messages": request.messages,
            "temperature": self.backend_config.temperature,
            "top_p": self.backend_config.top_p,
            "max_tokens": self.backend_config.max_tokens,
            "repetition_penalty": self.backend_config.repetition_penalty,
            "seed": self.backend_config.seed,
            "stream": False,
        }
        started = perf_counter()
        async with session.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"chat completions failed: HTTP {response.status}: {body[:300]}")
            result = await response.json()
        latency_ms = int((perf_counter() - started) * 1000)
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError("chat completions response has empty choices")
        text = str(choices[0].get("message", {}).get("content", "")).strip()
        return GenerationResponse(
            text=text,
            backend=self.persona_model,
            model_name=self.backend_config.model_name,
            latency_ms=latency_ms,
        )

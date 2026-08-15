"""OpenAI-compatible chat completions backend (local llama-server or remote API)."""

from __future__ import annotations

from time import perf_counter

import aiohttp

from src.core.llm.base import GenerationRequest, GenerationResponse
from src.core.settings.generation_config import BackendConfig


class ChatCompletionsBackend:
    def __init__(
        self,
        *,
        server_url: str,
        backend_config: BackendConfig,
        session: aiohttp.ClientSession | None = None,
        persona_model: str = "base_strong",
        api_key: str | None = None,
        spawn_local: bool = True,
    ):
        self.server_url = server_url.rstrip("/")
        self.backend_config = backend_config
        self.session = session
        self.persona_model = persona_model
        self.api_key = (api_key or "").strip() or None
        self.spawn_local = spawn_local
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

    def _completions_path(self) -> str:
        return "/v1/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, request: GenerationRequest) -> dict:
        payload: dict = {
            "model": self.backend_config.model_name,
            "messages": request.messages,
            "temperature": self.backend_config.temperature,
            "top_p": self.backend_config.top_p,
            "max_tokens": self.backend_config.max_tokens,
            "stream": False,
        }
        if self.spawn_local:
            payload["repetition_penalty"] = self.backend_config.repetition_penalty
            payload["seed"] = self.backend_config.seed
        return payload

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        session = await self._ensure_session()
        payload = self._build_payload(request=request)
        started = perf_counter()
        async with session.post(
            f"{self.server_url}{self._completions_path()}",
            json=payload,
            headers=self._build_headers(),
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(
                    f"chat completions failed: HTTP {response.status}: {body[:300]}"
                )
            result = await response.json()
        latency_ms = int((perf_counter() - started) * 1000)
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError("chat completions response has empty choices")
        choice = choices[0]
        text = str(choice.get("message", {}).get("content", "")).strip()
        finish_reason = choice.get("finish_reason")
        usage_raw = result.get("usage") or {}
        usage: dict[str, int] | None = None
        if isinstance(usage_raw, dict) and usage_raw:
            usage = {
                str(key): int(value)
                for key, value in usage_raw.items()
                if isinstance(value, (int, float))
            }
        return GenerationResponse(
            text=text,
            backend=self.persona_model,
            model_name=self.backend_config.model_name,
            latency_ms=latency_ms,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage,
        )

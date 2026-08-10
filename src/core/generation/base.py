"""Shared generation backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_content: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    backend: str
    model_name: str
    latency_ms: int
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class GenerationBackend(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate model output for a prepared request."""

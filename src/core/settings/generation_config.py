"""Generation backend configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
import yaml

from src.core.settings.deepseek_config import (
    DEEPSEEK_DEFAULT_MODEL,
    DEEPSEEK_DEFAULT_SERVER_URL,
    LOCAL_DEFAULT_SERVER_URL,
    validate_deepseek_payload,
)

PersonaModel = Literal["base_strong"]
ApiStyle = Literal["chat_completions"]
Provider = Literal["llama", "deepseek"]
ThinkingMode = Literal["enabled", "disabled"]
ReasoningEffort = Literal["low", "high", "max"]


class FallbackConfig(BaseModel):
    enabled: bool = False
    incident_threshold: int = 5
    window_events: int = 50
    audit_log_path: str = ".cursor/artifacts/safety/dryrun_audit.jsonl"


class SafetyConfig(BaseModel):
    post_filter: bool = True
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)


class BackendConfig(BaseModel):
    model_name: str
    model_path: str
    api_style: ApiStyle = "chat_completions"
    ctx_size: int = 4096
    n_gpu_layers: int = 28
    threads: int = 4
    temperature: float = 0.4
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    max_tokens: int = 300
    seed: int = 42
    max_context_chars: int = 3000
    thinking_mode: ThinkingMode = "disabled"
    reasoning_effort: ReasoningEffort | None = None


class GenerationConfig(BaseModel):
    persona_model: PersonaModel = "base_strong"
    provider: Provider = "llama"
    server_url: str = LOCAL_DEFAULT_SERVER_URL
    comparison_seed: int = 42
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    api_key: str | None = None
    spawn_local: bool = True

    def active_backend(self) -> BackendConfig:
        backend = self.backends.get(self.persona_model)
        if backend is None:
            raise ValueError(
                f"Missing backend config for persona_model={self.persona_model}"
            )
        return backend

    def with_persona_model(self, persona_model: PersonaModel) -> "GenerationConfig":
        payload = self.model_dump()
        payload["persona_model"] = persona_model
        return GenerationConfig.model_validate(payload)


def normalize_server_url(url: str) -> str:
    """Strip trailing slash and optional trailing /v1 to avoid /v1/v1/... joins."""
    normalized = (url or "").strip().rstrip("/")
    if normalized.lower().endswith("/v1"):
        normalized = normalized[: -len("/v1")].rstrip("/")
    return normalized


def llm_spawn_local_from_env() -> bool:
    raw = os.getenv("LLM_SPAWN_LOCAL", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def provider_from_env(default: Provider = "llama") -> Provider:
    raw = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not raw:
        return default
    if raw not in {"llama", "deepseek"}:
        raise ValueError(
            f"Invalid LLM_PROVIDER={raw!r}; expected 'llama' or 'deepseek'"
        )
    return raw  # type: ignore[return-value]


def _set_active_model_name(payload: dict, model_name: str) -> None:
    persona = payload["persona_model"]
    backends = dict(payload.get("backends") or {})
    active = dict(backends.get(persona) or {})
    if not active:
        raise ValueError(f"Missing backend config for persona_model={persona}")
    active["model_name"] = model_name
    backends[persona] = active
    payload["backends"] = backends


def apply_generation_env_overrides(config: GenerationConfig) -> GenerationConfig:
    """Apply VPS/remote env overrides. Defaults keep local Windows spawn behavior."""
    payload = config.model_dump()
    spawn_local = llm_spawn_local_from_env()
    payload["spawn_local"] = spawn_local
    provider = provider_from_env(default=payload.get("provider") or "llama")
    payload["provider"] = provider

    server_override = os.getenv("GENERATION_SERVER_URL", "").strip()
    if server_override:
        payload["server_url"] = normalize_server_url(server_override)
    elif provider == "deepseek":
        current = normalize_server_url(str(payload.get("server_url") or ""))
        if not current or current == LOCAL_DEFAULT_SERVER_URL:
            payload["server_url"] = DEEPSEEK_DEFAULT_SERVER_URL

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key and provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    payload["api_key"] = api_key or None

    model_override = os.getenv("LLM_MODEL_NAME", "").strip()
    if provider == "deepseek" and not model_override:
        model_override = DEEPSEEK_DEFAULT_MODEL
    if not spawn_local and not model_override and provider != "deepseek":
        raise ValueError(
            "LLM_MODEL_NAME is required when LLM_SPAWN_LOCAL=false (remote mode)"
        )
    if model_override:
        _set_active_model_name(payload, model_name=model_override)

    if provider == "deepseek":
        validate_deepseek_payload(
            payload=payload,
            normalize_server_url=normalize_server_url,
        )

    return GenerationConfig.model_validate(payload)


def load_generation_config(path: Path) -> GenerationConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("generation", payload)
    config = GenerationConfig.model_validate(section)
    return apply_generation_env_overrides(config)


def default_generation_config_path(base_dir: Path) -> Path:
    return base_dir / "config" / "generation.yaml"

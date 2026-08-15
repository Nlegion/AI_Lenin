"""Generation backend configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
import yaml


PersonaModel = Literal["base_strong"]
ApiStyle = Literal["chat_completions"]


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


class GenerationConfig(BaseModel):
    persona_model: PersonaModel = "base_strong"
    server_url: str = "http://127.0.0.1:8080"
    comparison_seed: int = 42
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)

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


def load_generation_config(path: Path) -> GenerationConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("generation", payload)
    return GenerationConfig.model_validate(section)


def default_generation_config_path(base_dir: Path) -> Path:
    return base_dir / "config" / "generation.yaml"

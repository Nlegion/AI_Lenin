"""Factory for generation backends."""

from __future__ import annotations

from pathlib import Path

import aiohttp

from src.core.generation.chat_backend import ChatCompletionsBackend
from src.core.generation.completion_backend import CompletionBackend
from src.core.generation.fallback import recommend_persona_model
from src.core.settings.generation_config import (
    GenerationConfig,
    PersonaModel,
    default_generation_config_path,
    load_generation_config,
)


def load_config(base_dir: Path, config_path: Path | None = None) -> GenerationConfig:
    path = config_path or default_generation_config_path(base_dir=base_dir)
    return load_generation_config(path=path)


def build_generation_backend(
    *,
    base_dir: Path,
    config: GenerationConfig | None = None,
    persona_model: PersonaModel | None = None,
    session: aiohttp.ClientSession | None = None,
    apply_fallback_recommendation: bool = False,
):
    cfg = config or load_config(base_dir=base_dir)
    selected = persona_model or cfg.persona_model
    if apply_fallback_recommendation:
        selected = recommend_persona_model(config=cfg, base_dir=base_dir)
    cfg = cfg.with_persona_model(selected)
    backend_cfg = cfg.active_backend()
    if backend_cfg.api_style == "chat_completions":
        return ChatCompletionsBackend(
            server_url=cfg.server_url,
            backend_config=backend_cfg,
            session=session,
            persona_model=selected,
        ), cfg
    return CompletionBackend(
        server_url=cfg.server_url,
        backend_config=backend_cfg,
        session=session,
        persona_model=selected,
    ), cfg

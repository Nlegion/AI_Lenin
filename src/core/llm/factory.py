"""Factory for LLM generation backends."""

from __future__ import annotations

from pathlib import Path

import aiohttp

from src.core.llm.chat_completions import ChatCompletionsBackend
from src.core.llm.deepseek import DeepSeekBackend
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
        # Lazy import keeps llm free of module-level generation deps.
        from src.core.generation.fallback import recommend_persona_model

        selected = recommend_persona_model(config=cfg, base_dir=base_dir)
    cfg = cfg.with_persona_model(selected)
    backend_cfg = cfg.active_backend()
    if backend_cfg.api_style != "chat_completions":
        raise ValueError(
            f"Unsupported api_style={backend_cfg.api_style!r}; only chat_completions is supported"
        )
    backend_cls = (
        DeepSeekBackend if cfg.provider == "deepseek" else ChatCompletionsBackend
    )
    return backend_cls(
        server_url=cfg.server_url,
        backend_config=backend_cfg,
        session=session,
        persona_model=selected,
        api_key=cfg.api_key,
        spawn_local=cfg.spawn_local,
    ), cfg

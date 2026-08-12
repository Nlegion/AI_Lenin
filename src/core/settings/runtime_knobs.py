"""Resolve overlapping generation knobs with generation.yaml as source of truth."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.core.dialectics.config import DialecticalReasoningConfig, load_dialectical_reasoning_config
from src.core.settings.generation_config import default_generation_config_path, load_generation_config


def load_reasoning_config_with_generation_sot(
    *,
    base_dir: Path,
) -> DialecticalReasoningConfig:
    """Prefer generation.yaml for shared LLM runtime knobs.

    Ownership:
    - retrieval_pipeline.yaml → retrieval + orchestration structure
    - generation.yaml → ctx_size / temperature / max_tokens (SoT)
    - dialectical_reasoning section → reasoning-specific limits only
    """
    reasoning = load_dialectical_reasoning_config(base_dir=base_dir)
    gen_path = default_generation_config_path(base_dir)
    if not gen_path.is_file():
        return reasoning
    generation = load_generation_config(path=gen_path)
    backend = generation.active_backend()
    return replace(
        reasoning,
        ctx_size=int(backend.ctx_size),
        temperature=float(backend.temperature),
        max_tokens_out=int(backend.max_tokens),
    )

"""Runtime config loader for standalone pre-RAG censor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.core.safety.pre_rag_censor import CensorRuntimeConfig


def default_censorship_runtime_config_path(base_dir: Path) -> Path:
    return base_dir / "config" / "safety_gate_config.yaml"


def load_censorship_runtime_config(path: Path) -> CensorRuntimeConfig:
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("safety_gate", payload)
    runtime = section.get("censorship_runtime") or {}
    return CensorRuntimeConfig(**runtime)

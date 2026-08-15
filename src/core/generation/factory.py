"""Compatibility shim. Prefer: from src.core.llm.factory import build_generation_backend"""

from src.core.llm.factory import build_generation_backend, load_config

__all__ = ["build_generation_backend", "load_config"]

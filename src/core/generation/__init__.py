"""Generation backends and safety-aware analysis pipeline.

Lazy exports to avoid circular imports with dialectics and quality modules.
"""

from __future__ import annotations

from typing import Any

__all__ = ["AnalysisGenerationPipeline", "build_generation_backend"]


def __getattr__(name: str) -> Any:
    if name == "AnalysisGenerationPipeline":
        from src.core.generation.pipeline import AnalysisGenerationPipeline

        return AnalysisGenerationPipeline
    if name == "build_generation_backend":
        from src.core.llm.factory import build_generation_backend

        return build_generation_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Compatibility shim. Prefer: from src.core.llm.base import ..."""

from src.core.llm.base import GenerationBackend, GenerationRequest, GenerationResponse

__all__ = ["GenerationBackend", "GenerationRequest", "GenerationResponse"]

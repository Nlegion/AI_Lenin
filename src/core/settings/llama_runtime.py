"""Compatibility shim. Prefer: from src.core.llm.runtime import ..."""

from src.core.llm.runtime import LlamaRuntimePaths, resolve_llama_runtime

__all__ = ["LlamaRuntimePaths", "resolve_llama_runtime"]

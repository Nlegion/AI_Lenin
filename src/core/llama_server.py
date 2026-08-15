"""Compatibility shim. Prefer: from src.core.llm.server import LeninServer"""

from src.core.llm.server import LeninServer

__all__ = ["LeninServer"]

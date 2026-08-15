"""Shared contracts for retrieval providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RetrievalResult:
    """Normalized context result for downstream prompt builder."""

    context: str
    candidates_count: int
    metadata: dict[str, str] = field(default_factory=dict)


class RetrievalProvider(Protocol):
    """Provider contract used by the analyzer runtime."""

    def retrieve_context(
        self, query_text: str, author_filter: str | None = None
    ) -> RetrievalResult:
        """Return normalized context and minimal retrieval metadata."""

"""Quality checks for chunking v2 outputs."""

from __future__ import annotations

from src.core.preprocessing.chunker_v2 import ChunkRecord


def bad_boundary_ratio(chunks: list[ChunkRecord]) -> float:
    if not chunks:
        return 0.0
    bad = sum(1 for chunk in chunks if not chunk.boundary_ok)
    return bad / len(chunks)


def token_window_compliance_ratio(
    chunks: list[ChunkRecord], min_tokens: int, max_tokens: int
) -> float:
    if not chunks:
        return 1.0
    compliant = sum(
        1 for chunk in chunks if min_tokens <= chunk.token_count <= max_tokens
    )
    return compliant / len(chunks)

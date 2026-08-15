"""Extract quote candidates from trusted RAG chunks (never from news body).

Chunk metadata keys for attribution: author, work/title, volume/том, page/стр,
source_id, chunk_id, optional quote/thesis (explicit cite-marked spans).
See docs/quote_grounding.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.core.generation.text_normalize import normalize_for_grounding
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

# Paired quote spans inside one chunk.
_QUOTE_SPAN = re.compile(
    r"«([^»]{3,400})»|"
    r"\"([^\"]{3,400})\"|"
    r"„([^“]{3,400})“|"
    r"“([^”]{3,400})”"
)
_CONTENT_TOKEN = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class QuoteCandidate:
    text: str
    chunk_id: str
    source_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _content_token_count(text: str) -> int:
    return len(_CONTENT_TOKEN.findall(text))


def _is_trivial(text: str, stoplist: list[str]) -> bool:
    stripped = text.strip()
    for pattern in stoplist:
        if re.search(pattern, stripped):
            return True
    return False


def _meta_from_chunk(meta: dict[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    keys = (
        "author",
        "work",
        "title",
        "volume",
        "том",
        "page",
        "стр",
        "page_start",
        "source_id",
        "quote",
        "thesis",
    )
    return {k: meta[k] for k in keys if k in meta and meta[k] not in (None, "")}


def extract_quote_candidates(
    *,
    chunks: list[tuple[str, float, str]],
    chunk_meta: dict[str, dict[str, Any]] | None = None,
    config: QualityPostcheckConfig,
) -> list[QuoteCandidate]:
    """Build allowlist from RAG chunk texts + optional explicit meta spans."""
    meta_by_id = chunk_meta or {}
    out: list[QuoteCandidate] = []
    seen: set[str] = set()
    for chunk_id, _score, text in chunks:
        meta = _meta_from_chunk(meta_by_id.get(chunk_id))
        source_id = str(meta.get("source_id") or "") or None
        spans: list[str] = []
        for match in _QUOTE_SPAN.finditer(text or ""):
            span = next((g for g in match.groups() if g), None)
            if span:
                spans.append(span.strip())
        for key in ("quote", "thesis"):
            raw = meta.get(key)
            if isinstance(raw, str) and raw.strip():
                spans.append(raw.strip())
        for span in spans:
            if _is_trivial(span, config.trivial_quote_stoplist):
                continue
            if len(span) < config.min_quote_chars:
                continue
            if _content_token_count(span) < config.min_quote_content_tokens:
                continue
            norm = normalize_for_grounding(span)
            key = f"{chunk_id}:{norm}"
            if not norm or key in seen:
                continue
            seen.add(key)
            out.append(
                QuoteCandidate(
                    text=span,
                    chunk_id=chunk_id,
                    source_id=source_id,
                    meta=meta,
                )
            )
    return out


def quote_allowlist_present(candidates: list[QuoteCandidate]) -> bool:
    return bool(candidates)


def usable_for_context(
    chunks: list[tuple[str, float, str]], *, min_chars: int = 40
) -> bool:
    return any(len((text or "").strip()) >= min_chars for _cid, _s, text in chunks)


def usable_for_attribution(candidates: list[QuoteCandidate]) -> bool:
    for item in candidates:
        meta = item.meta
        if any(
            meta.get(k)
            for k in ("volume", "том", "page", "стр", "work", "title", "author")
        ):
            return True
    return False

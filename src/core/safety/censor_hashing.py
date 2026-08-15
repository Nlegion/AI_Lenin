"""Canonical hashing utilities for pre-RAG censorship."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[!?,.:;\-_=+~`^\"'()\[\]{}<>|/\\]+")
_TRACKING_PREFIXES = ("utm_", "yclid", "fbclid", "gclid")

NORMALIZER_VERSION = "norm-v2-nfkc-urlcanon-e2e"


def canonicalize_url(raw_url: str) -> str:
    """Canonicalize URL for dedup without dropping semantic path/query."""
    value = (raw_url or "").strip()
    if not value:
        return ""
    split = urlsplit(value)
    query_pairs = [
        (key, val)
        for key, val in parse_qsl(split.query, keep_blank_values=True)
        if not any(key.lower().startswith(prefix) for prefix in _TRACKING_PREFIXES)
    ]
    query_pairs.sort()
    netloc = split.netloc.lower()
    path = split.path or "/"
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((split.scheme.lower(), netloc, path, query, ""))


def normalize_for_content_hash(*, title: str, body: str, url: str = "") -> str:
    """Strict normalization pipeline for content hash."""
    merged = f"{title}\n{body}\n{canonicalize_url(url)}"
    step_1 = html.unescape(merged)
    step_2 = _TAG_RE.sub(" ", step_1)
    step_3 = unicodedata.normalize("NFKC", step_2)
    step_4 = step_3.lower().replace("ё", "е")
    step_5 = _TRAILING_PUNCT_RE.sub(" ", step_4)
    return _WS_RE.sub(" ", step_5).strip()


def compute_content_hash(*, title: str, body: str, url: str = "") -> tuple[str, str]:
    normalized = normalize_for_content_hash(title=title, body=body, url=url)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest, normalized


def canonical_json_hash(payload: Any) -> str:
    """Stable hash for nested structures with deterministic ordering."""
    if is_dataclass(payload):
        value = asdict(payload)
    else:
        value = payload
    dumped = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

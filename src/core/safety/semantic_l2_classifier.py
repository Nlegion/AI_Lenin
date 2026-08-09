"""Lightweight semantic L2 classifier abstraction."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class L2Score:
    category: str
    score: float


class SemanticL2Classifier:
    """Keyword baseline wrapper with runtime-bounded interface."""

    def __init__(
        self,
        *,
        model_version: str,
        prototypes: dict[str, tuple[str, ...]],
        text_max_chars: int = 1200,
        cache_size: int = 4096,
    ):
        self.model_version = model_version
        self._prototypes = prototypes
        self._text_max_chars = max(128, int(text_max_chars))
        self._cache_size = max(128, int(cache_size))
        self._cache: OrderedDict[str, L2Score] = OrderedDict()

    def _prepare_text(self, text: str) -> str:
        normalized = _WS_RE.sub(" ", text.lower()).strip()
        return normalized[: self._text_max_chars]

    def _cache_get(self, key: str) -> L2Score | None:
        value = self._cache.get(key)
        if value is None:
            return None
        self._cache.move_to_end(key)
        return value

    def _cache_set(self, key: str, value: L2Score) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def score(self, text: str) -> L2Score:
        prepared = self._prepare_text(text)
        key = hashlib.sha1(prepared.encode("utf-8")).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        best_category = "SENSITIVE_POLITICS"
        best_score = 0.0
        for category, stems in self._prototypes.items():
            hits = sum(1 for stem in stems if stem in prepared)
            score = float(hits / max(len(stems), 1))
            if score > best_score:
                best_score = score
                best_category = category
        result = L2Score(category=best_category, score=best_score)
        self._cache_set(key, result)
        return result


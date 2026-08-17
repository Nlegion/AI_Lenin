"""In-process windowed counters for admin ops digests."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, MutableMapping


_COUNTER_KEYS = (
    "rss_seen",
    "dedup_dropped",
    "inserted",
    "news_processed",
    "news_skipped",
    "review_continued",
    "output_guard_blocked",
    "analyses_rejected",
    "analyses_published",
    "publish_failed",
    "errors",
    "degraded_held",
    "generation_timeouts",
    "circuit_opens",
)

# Map legacy pipeline keys onto the funnel names used in digests.
_LEGACY_ALIASES = {
    "news_fetched": "rss_seen",
}


@dataclass
class WindowSnapshot:
    """Frozen view of one report window (before reset)."""

    counters: dict[str, int]
    skip_reasons: Counter[str]
    reject_reasons: Counter[str]
    latency_samples_ms: list[int]
    started_at: float
    window_started_at: float
    llm_provider: str = ""

    @property
    def db_duplicates(self) -> int:
        raw = self.counters.get("rss_seen", 0) - self.counters.get("dedup_dropped", 0)
        inserted = self.counters.get("inserted", 0)
        return max(0, raw - inserted)

    def percentile(self, pct: float) -> int | None:
        samples = sorted(self.latency_samples_ms)
        if not samples:
            return None
        if len(samples) == 1:
            return samples[0]
        index = int(round((pct / 100.0) * (len(samples) - 1)))
        index = max(0, min(len(samples) - 1, index))
        return samples[index]

    def uptime_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def is_idle(self) -> bool:
        activity = (
            self.counters.get("inserted", 0)
            + self.counters.get("news_processed", 0)
            + self.counters.get("news_skipped", 0)
            + self.counters.get("analyses_published", 0)
            + self.counters.get("analyses_rejected", 0)
            + self.counters.get("publish_failed", 0)
            + self.counters.get("errors", 0)
            + self.counters.get("review_continued", 0)
            + self.counters.get("output_guard_blocked", 0)
        )
        return activity == 0


class WindowStats(MutableMapping[str, int]):
    """Dict-like window counters plus reason histograms and latency samples."""

    def __init__(
        self,
        *,
        max_latency_samples: int = 50,
        llm_provider: str = "",
    ) -> None:
        self._max_latency_samples = max(1, int(max_latency_samples))
        self.llm_provider = llm_provider
        self.started_at = time.time()
        self.window_started_at = self.started_at
        self._counters: dict[str, int] = {key: 0 for key in _COUNTER_KEYS}
        self.skip_reasons: Counter[str] = Counter()
        self.reject_reasons: Counter[str] = Counter()
        self.latency_samples_ms: list[int] = []
        self._circuit_baseline_timeouts = 0
        self._circuit_baseline_opens = 0

    def __getitem__(self, key: str) -> int:
        return self._counters[self._resolve(key)]

    def __setitem__(self, key: str, value: int) -> None:
        self._counters[self._resolve(key)] = int(value)

    def __delitem__(self, key: str) -> None:
        resolved = self._resolve(key)
        if resolved not in self._counters:
            raise KeyError(key)
        self._counters[resolved] = 0

    def __iter__(self) -> Iterator[str]:
        return iter(self._counters)

    def __len__(self) -> int:
        return len(self._counters)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._resolve(key) in self._counters

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        resolved = self._resolve(key)
        if resolved not in self._counters:
            return default
        return self._counters[resolved]

    def keys(self):  # type: ignore[override]
        return self._counters.keys()

    def items(self):  # type: ignore[override]
        return self._counters.items()

    def values(self):  # type: ignore[override]
        return self._counters.values()

    @staticmethod
    def _resolve(key: str) -> str:
        return _LEGACY_ALIASES.get(key, key)

    def record_skip_reasons(self, codes: list[str] | None) -> None:
        if not codes:
            self.skip_reasons["unspecified"] += 1
            return
        for code in codes:
            token = str(code or "").strip() or "unspecified"
            self.skip_reasons[token] += 1

    def record_reject_reasons(self, reasons: list[str] | None) -> None:
        if not reasons:
            self.reject_reasons["unspecified"] += 1
            return
        for reason in reasons:
            token = str(reason or "").strip() or "unspecified"
            self.reject_reasons[token] += 1

    def record_latency_ms(self, latency_ms: int | None) -> None:
        if latency_ms is None:
            return
        value = int(latency_ms)
        if value < 0:
            return
        self.latency_samples_ms.append(value)
        overflow = len(self.latency_samples_ms) - self._max_latency_samples
        if overflow > 0:
            del self.latency_samples_ms[:overflow]

    def sync_circuit_deltas(
        self,
        *,
        total_timeouts: int,
        total_opens: int,
    ) -> None:
        """Apply lifetime circuit totals as window deltas (not overwrites)."""
        timeouts = max(0, int(total_timeouts) - self._circuit_baseline_timeouts)
        opens = max(0, int(total_opens) - self._circuit_baseline_opens)
        if timeouts:
            self._counters["generation_timeouts"] += timeouts
            self._circuit_baseline_timeouts = int(total_timeouts)
        if opens:
            self._counters["circuit_opens"] += opens
            self._circuit_baseline_opens = int(total_opens)

    def snapshot(self) -> WindowSnapshot:
        return WindowSnapshot(
            counters=dict(self._counters),
            skip_reasons=Counter(self.skip_reasons),
            reject_reasons=Counter(self.reject_reasons),
            latency_samples_ms=list(self.latency_samples_ms),
            started_at=self.started_at,
            window_started_at=self.window_started_at,
            llm_provider=self.llm_provider,
        )

    def snapshot_and_reset(self) -> WindowSnapshot:
        snap = self.snapshot()
        for key in self._counters:
            self._counters[key] = 0
        self.skip_reasons.clear()
        self.reject_reasons.clear()
        self.latency_samples_ms.clear()
        self.window_started_at = time.time()
        return snap

    def as_mapping(self) -> Mapping[str, int]:
        return dict(self._counters)

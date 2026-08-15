"""Load release_gates.yaml (unified RAG thresholds + gate toggles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MetricThreshold:
    threshold: float
    direction: str  # higher | lower


@dataclass(frozen=True)
class RagQualityGate:
    enabled: bool = True
    tolerance_relative: float = 0.03
    metrics: dict[str, MetricThreshold] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseGatesConfig:
    version: str = "1.0"
    rag_quality: RagQualityGate = field(default_factory=RagQualityGate)
    news_guard_enabled: bool = True
    anti_cliche_enabled: bool = False
    news_guard_delta_enabled: bool = False
    news_guard_baseline_json: str = (
        ".cursor/artifacts/evaluation/news_guard_eval_baseline.json"
    )


def _parse_metrics(raw: dict[str, Any] | None) -> dict[str, MetricThreshold]:
    if not raw:
        return {}
    metrics: dict[str, MetricThreshold] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            direction = str(value.get("direction", "higher"))
            metrics[name] = MetricThreshold(
                threshold=float(value["threshold"]),
                direction=direction,
            )
        else:
            # Legacy flat float shim
            direction = "lower" if str(name).endswith("_rate_max") else "higher"
            metrics[name] = MetricThreshold(threshold=float(value), direction=direction)
    return metrics


def metric_passes(
    *,
    value: float,
    threshold: float,
    direction: str,
    tolerance_relative: float,
) -> bool:
    if direction == "lower":
        limit = threshold * (1.0 + tolerance_relative)
        return value <= limit
    limit = threshold * (1.0 - tolerance_relative)
    return value >= limit


@lru_cache(maxsize=4)
def load_release_gates(path: str | None = None) -> ReleaseGatesConfig:
    config_path = Path(path) if path else _REPO_ROOT / "config" / "release_gates.yaml"
    if not config_path.is_file():
        return ReleaseGatesConfig()

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    # Support accidental nesting under release_gates:
    if "rag_quality" not in payload and "release_gates" in payload:
        payload = payload["release_gates"]
    rag_raw = payload.get("rag_quality") or {}
    metrics_raw = rag_raw.get("metrics")
    rag = RagQualityGate(
        enabled=bool(rag_raw.get("enabled", True)),
        tolerance_relative=float(rag_raw.get("tolerance_relative", 0.03)),
        metrics=_parse_metrics(metrics_raw),
    )
    news = payload.get("news_guard") or {}
    anti = payload.get("anti_cliche") or {}
    delta = payload.get("news_guard_delta_check") or {}
    return ReleaseGatesConfig(
        version=str(payload.get("version", "1.0")),
        rag_quality=rag,
        news_guard_enabled=bool(news.get("enabled", True)),
        anti_cliche_enabled=bool(anti.get("enabled", False)),
        news_guard_delta_enabled=bool(delta.get("enabled", False)),
        news_guard_baseline_json=str(
            delta.get(
                "baseline_json",
                ".cursor/artifacts/evaluation/news_guard_eval_baseline.json",
            )
        ),
    )

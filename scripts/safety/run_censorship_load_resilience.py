"""Run load and resilience checks for pre-RAG censor."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from src.core.safety.pre_rag_censor import CensorRuntimeConfig, PreRagCensor
from src.core.safety.pre_rag_censor_types import CensorInput
from src.core.safety.safety_gate import SafetyGate
from src.core.safety.news_guard import NewsGuard
from src.core.settings.censorship_runtime_config import (
    default_censorship_runtime_config_path,
    load_censorship_runtime_config,
)
from src.modules.news_system.fetcher import NewsFetcher


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(p * (len(arr) - 1))
    return arr[idx]


async def _run_batch(
    censor: PreRagCensor, items: list[dict[str, Any]]
) -> dict[str, float]:
    latencies: list[float] = []
    failures = 0
    started = time.perf_counter()
    for item in items:
        try:
            result = await censor.evaluate(
                CensorInput(
                    news_id=str(item.get("id")),
                    title=str(item.get("title") or ""),
                    body=str(item.get("content") or ""),
                    source=str(item.get("source") or "unknown"),
                    metadata={"url": str(item.get("url") or "")},
                )
            )
            latencies.append(float(result.audit.get("latency_ms", 0.0)))
        except Exception:
            failures += 1
    elapsed = max(time.perf_counter() - started, 1e-6)
    return {
        "items": float(len(items)),
        "failures": float(failures),
        "failure_rate": float(failures / max(len(items), 1)),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "throughput_items_per_second": float(len(items) / elapsed),
    }


def _build_censor(*, base_dir: Path, mode: str) -> PreRagCensor:
    gate = SafetyGate.from_base_dir(base_dir)
    guard = NewsGuard.from_file(base_dir / "config" / "news_guard.yaml")
    cfg_path = default_censorship_runtime_config_path(base_dir)
    runtime = load_censorship_runtime_config(cfg_path)
    if mode == "l2_off":
        runtime = CensorRuntimeConfig(
            **{**runtime.__dict__, "l2_similarity_enabled": False}
        )
    return PreRagCensor(
        safety_gate=gate,
        news_guard=guard,
        config=runtime,
        config_path=str(cfg_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json",
        default=".cursor/artifacts/quality/censorship_load_resilience_latest.json",
    )
    parser.add_argument("--items", type=int, default=600)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    source_items = NewsFetcher().fetch_all()
    if not source_items:
        raise RuntimeError("No live news fetched for load/resilience check")
    items = (source_items * ((args.items // len(source_items)) + 1))[: args.items]

    baseline = asyncio.run(
        _run_batch(_build_censor(base_dir=base_dir, mode="baseline"), items)
    )
    l2_off = asyncio.run(
        _run_batch(_build_censor(base_dir=base_dir, mode="l2_off"), items)
    )
    report = {"baseline": baseline, "l2_off": l2_off}

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

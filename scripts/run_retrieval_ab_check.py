#!/usr/bin/env python3
"""Run retrieval parity checks between primary and shadow providers."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.rag_system import get_rag_system  # noqa: E402
from src.core.retrieval.provider_factory import build_provider  # noqa: E402


def _load_queries(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return [
            "Санкции и инфляция усилились в стране",
            "Рост безработицы и сокращение зарплат",
            "Решение правительства о торговых пошлинах",
            "Международный кризис и конкуренция капиталов",
            "Социальные протесты рабочих и профсоюзов",
        ]
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval A/B parity check.")
    parser.add_argument("--config", default="config/retrieval_pipeline.yaml")
    parser.add_argument("--queries", default=None, help="Optional text file with one query per line.")
    parser.add_argument("--out-json", default=".cursor/artifacts/retrieval/retrieval_ab_summary.json")
    parser.add_argument("--out-md", default=".cursor/artifacts/retrieval/retrieval_ab_summary.md")
    args = parser.parse_args()

    provider = build_provider(
        config_path=(REPO_ROOT / args.config).resolve(),
        base_dir=REPO_ROOT,
        rag_system=get_rag_system(),
    )
    if provider is None:
        print("Retrieval provider is disabled by config.")
        return 1

    queries = _load_queries(path=(REPO_ROOT / args.queries).resolve() if args.queries else None)
    trace: list[dict[str, str | int | float | bool]] = []
    parity_values: list[float] = []
    non_empty = 0
    for query in queries:
        result = provider.retrieve_context(query_text=query, author_filter="Ленин")
        if result.context.strip():
            non_empty += 1
        parity_value = _to_float(result.metadata.get("parity_shared_ratio"))
        if parity_value is not None:
            parity_values.append(parity_value)
        trace.append(
            {
                "query": query,
                "context_chars": len(result.context),
                "candidates_count": result.candidates_count,
                "provider": result.metadata.get("provider", "unknown"),
                "mode": result.metadata.get("mode", "single"),
                "parity_shared_ratio": result.metadata.get("parity_shared_ratio", ""),
            }
        )

    payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "queries_total": len(queries),
        "non_empty_context_rate": non_empty / len(queries) if queries else 0.0,
        "parity_shared_ratio_avg": statistics.mean(parity_values) if parity_values else None,
        "trace": trace,
    }

    out_json = (REPO_ROOT / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = (REPO_ROOT / args.out_md).resolve()
    lines = [
        "# Retrieval A/B Summary",
        "",
        f"- Generated at (UTC): {payload['generated_at_utc']}",
        f"- Queries total: `{payload['queries_total']}`",
        f"- Non-empty context rate: `{payload['non_empty_context_rate']:.3f}`",
        f"- Average parity shared ratio: `{payload['parity_shared_ratio_avg']}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Queries total: {payload['queries_total']}")
    print(f"Non-empty context rate: {payload['non_empty_context_rate']:.3f}")
    print(f"Average parity shared ratio: {payload['parity_shared_ratio_avg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

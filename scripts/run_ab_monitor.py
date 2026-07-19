#!/usr/bin/env python3
"""Pre-cutover A/B monitor simulation for retrieval parity."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.rag_system import get_rag_system  # noqa: E402
from src.core.retrieval.provider_factory import build_provider  # noqa: E402


def _queries() -> list[str]:
    return [
        "Рост инфляции и безработицы",
        "Санкции и международная торговля",
        "Социальные протесты рабочих",
        "Политика правительства и бюджет",
        "Империализм и мировой рынок",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-cutover A/B parity monitor.")
    parser.add_argument("--config", default="config/retrieval_pipeline.yaml")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out-json", default=".cursor/artifacts/retrieval/ab_monitor.json")
    parser.add_argument("--out-md", default=".cursor/artifacts/retrieval/ab_monitor.md")
    args = parser.parse_args()

    provider = build_provider(
        config_path=(REPO_ROOT / args.config).resolve(),
        base_dir=REPO_ROOT,
        rag_system=get_rag_system(),
    )
    if provider is None:
        raise RuntimeError("Retrieval provider disabled in config.")

    rounds: list[dict[str, float]] = []
    base_queries = _queries()
    for round_index in range(args.rounds):
        queries = base_queries[:]
        random.Random(round_index).shuffle(queries)
        parity_values: list[float] = []
        non_empty = 0
        for query in queries:
            result = provider.retrieve_context(query_text=query, author_filter="Ленин")
            if result.context.strip():
                non_empty += 1
            parity_raw = result.metadata.get("parity_shared_ratio")
            if parity_raw is not None:
                parity_values.append(float(parity_raw))
        rounds.append(
            {
                "round": round_index + 1,
                "queries_total": len(queries),
                "non_empty_ratio": non_empty / len(queries),
                "parity_mean": statistics.mean(parity_values) if parity_values else 0.0,
            }
        )

    parity_means = [row["parity_mean"] for row in rounds]
    payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rounds": rounds,
        "parity_mean_global": statistics.mean(parity_means) if parity_means else 0.0,
        "parity_min_round": min(parity_means) if parity_means else 0.0,
        "parity_max_round": max(parity_means) if parity_means else 0.0,
    }

    out_json = (REPO_ROOT / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = (REPO_ROOT / args.out_md).resolve()
    lines = [
        "# A/B Monitor Summary",
        "",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        f"- Rounds: `{len(rounds)}`",
        f"- Global parity mean: `{payload['parity_mean_global']:.4f}`",
        f"- Parity min round: `{payload['parity_min_round']:.4f}`",
        f"- Parity max round: `{payload['parity_max_round']:.4f}`",
        "",
        "| Round | Non-empty ratio | Parity mean |",
        "|---|---:|---:|",
    ]
    for row in rounds:
        lines.append(
            f"| {int(row['round'])} | {row['non_empty_ratio']:.3f} | {row['parity_mean']:.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ab_monitor_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

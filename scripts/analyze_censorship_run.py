"""Analyze censorship run artifacts and emit markdown report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _top(counter: Counter, n: int = 10) -> list[tuple[str, int]]:
    return [(str(k), int(v)) for k, v in counter.most_common(n)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.jsonl))
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    decision_counts = Counter(str(r.get("decision") or "unknown") for r in rows)
    category_counts = Counter(str(r.get("category") or "None") for r in rows)
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(str(x) for x in (row.get("reason_codes") or []))
    source_counts = Counter(str(r.get("source") or "unknown") for r in rows)
    dataset_counts = Counter(str(r.get("dataset") or "live") for r in rows)

    report = [
        "# Censorship Run Analysis",
        "",
        "## Summary",
        f"- rows: {len(rows)}",
        f"- hard_block_rate: {_pct(float(metrics.get('hard_block_rate', 0.0)))}",
        f"- review_rate: {_pct(float(metrics.get('review_rate', 0.0)))}",
        f"- skip_rate: {_pct(float(metrics.get('skip_rate', 0.0)))}",
        f"- allow_rate: {_pct(float(metrics.get('allow_rate', 0.0)))}",
        f"- reason_coverage: {_pct(float(metrics.get('reason_coverage', 0.0)))}",
        f"- p50_latency_ms: {float(metrics.get('p50_latency_ms', 0.0)):.2f}",
        f"- p95_latency_ms: {float(metrics.get('p95_latency_ms', 0.0)):.2f}",
        "",
        "## Policy Core Slice",
        f"- n_policy_core: {int(metrics.get('n_policy_core', 0))}",
        f"- policy_core_share: {_pct(float(metrics.get('policy_core_share', 0.0)))}",
        "",
        "## Decision Distribution",
    ]
    for key, value in decision_counts.items():
        report.append(f"- {key}: {value}")

    report.extend(["", "## Top Categories"])
    for key, value in _top(category_counts, n=12):
        report.append(f"- {key}: {value}")

    report.extend(["", "## Top Reason Codes"])
    for key, value in _top(reason_counts, n=15):
        report.append(f"- {key}: {value}")

    report.extend(["", "## Sources", *[f"- {k}: {v}" for k, v in _top(source_counts, n=10)]])
    report.extend(["", "## Dataset Split", *[f"- {k}: {v}" for k, v in _top(dataset_counts, n=10)]])

    Path(args.out).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"report={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

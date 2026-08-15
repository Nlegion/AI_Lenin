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


def _rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = max(len(rows), 1)
    decision_counts = Counter(str(r.get("decision") or "unknown") for r in rows)
    return {
        "hard_block_rate": decision_counts.get("hard_block", 0) / total,
        "review_rate": decision_counts.get("review", 0) / total,
        "skip_rate": decision_counts.get("skip", 0) / total,
        "allow_rate": decision_counts.get("allow", 0) / total,
        "reason_coverage": sum(1 for r in rows if r.get("reason_codes")) / total,
    }


def _first_unique_by_news_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        news_id = str(row.get("news_id") or "").strip()
        if not news_id or news_id in seen:
            continue
        seen.add(news_id)
        unique.append(row)
    return unique


def _emit_rate_block(
    report: list[str], *, title: str, rows: list[dict[str, Any]]
) -> None:
    rates = _rates(rows=rows)
    report.extend(
        [
            "",
            title,
            f"- rows: {len(rows)}",
            f"- hard_block_rate: {_pct(float(rates.get('hard_block_rate', 0.0)))}",
            f"- review_rate: {_pct(float(rates.get('review_rate', 0.0)))}",
            f"- skip_rate: {_pct(float(rates.get('skip_rate', 0.0)))}",
            f"- allow_rate: {_pct(float(rates.get('allow_rate', 0.0)))}",
            f"- reason_coverage: {_pct(float(rates.get('reason_coverage', 0.0)))}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare-jsonl", default=None)
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
    live_rows = [r for r in rows if str(r.get("dataset") or "live") != "control"]
    control_rows = [r for r in rows if str(r.get("dataset") or "live") == "control"]
    control_unique = _first_unique_by_news_id(rows=control_rows)
    l1_mismatch_rows = [
        r
        for r in rows
        if str(r.get("l1_decision") or "").strip()
        and str(r.get("l1_decision")) != str(r.get("decision"))
    ]
    intentional_override_rows = [
        r
        for r in l1_mismatch_rows
        if "unknown_topic_low_signal_allow_forward"
        in [str(c) for c in (r.get("reason_codes") or [])]
        or "override:unknown_topic_forward_trusted_source"
        in [str(c) for c in (r.get("reason_codes") or [])]
    ]
    unexpected_conflict_rows = [
        r for r in l1_mismatch_rows if r not in intentional_override_rows
    ]

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

    _emit_rate_block(report, title="## Live Decision Rates", rows=live_rows)
    _emit_rate_block(
        report, title="## Control Decision Rates (All Rows)", rows=control_rows
    )
    _emit_rate_block(
        report, title="## Control Decision Rates (Unique news_id)", rows=control_unique
    )

    l1_counts = Counter(
        f"{str(r.get('l1_decision'))}->{str(r.get('decision'))}"
        for r in l1_mismatch_rows
    )
    report.extend(
        ["", "## L1 To Final Mismatches", f"- mismatch_rows: {len(l1_mismatch_rows)}"]
    )
    for key, value in _top(l1_counts, n=10):
        report.append(f"- {key}: {value}")
    report.extend(
        [
            f"- intentional_override_rows: {len(intentional_override_rows)}",
            f"- unexpected_conflict_rows: {len(unexpected_conflict_rows)}",
        ]
    )

    report.extend(
        ["", "## Sources", *[f"- {k}: {v}" for k, v in _top(source_counts, n=10)]]
    )
    report.extend(
        [
            "",
            "## Dataset Split",
            *[f"- {k}: {v}" for k, v in _top(dataset_counts, n=10)],
        ]
    )

    if args.compare_jsonl:
        compare_rows = _load_jsonl(Path(args.compare_jsonl))
        compare_control = [
            r for r in compare_rows if str(r.get("dataset") or "live") == "control"
        ]
        compare_control_unique = _first_unique_by_news_id(rows=compare_control)
        compare_live = [
            r for r in compare_rows if str(r.get("dataset") or "live") != "control"
        ]
        report.extend(
            ["", "## Compare Baseline", f"- compare_jsonl: {args.compare_jsonl}"]
        )
        _emit_rate_block(report, title="### Current Live (Unique N/A)", rows=live_rows)
        _emit_rate_block(
            report, title="### Compare Live (Unique N/A)", rows=compare_live
        )
        _emit_rate_block(
            report, title="### Current Control Unique", rows=control_unique
        )
        _emit_rate_block(
            report, title="### Compare Control Unique", rows=compare_control_unique
        )
        report.extend(
            [
                "",
                "### Compare Sample Sizes",
                f"- current_control_rows_all: {len(control_rows)}",
                f"- compare_control_rows_all: {len(compare_control)}",
                f"- current_control_unique: {len(control_unique)}",
                f"- compare_control_unique: {len(compare_control_unique)}",
            ]
        )

    Path(args.out).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"report={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

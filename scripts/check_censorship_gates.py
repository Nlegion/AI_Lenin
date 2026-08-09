"""Evaluate replay metrics against release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from _external_dataset_prestep import ensure_external_dataset_prestep


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def _decision(row: dict[str, Any]) -> str:
    return str(row.get("new_decision") or row.get("decision") or "")


def _reason_codes(row: dict[str, Any]) -> list[Any]:
    return list(row.get("new_reason_codes") or row.get("reason_codes") or [])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_or_load_latency_baseline(path: Path, current: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if path.is_file():
        return _load_json(path), False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--gates-config", default="config/release_gates.yaml")
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional metrics JSON from isolated run for latency/throughput gates.",
    )
    parser.add_argument(
        "--latency-baseline-json",
        default=".cursor/artifacts/quality/censorship_latency_baseline.json",
        help="Baseline snapshot for relative latency/throughput gate checks.",
    )
    parser.add_argument(
        "--latency-mode",
        choices=["auto", "live", "replay"],
        default="auto",
        help="Latency gate mode. replay skips relative p95/throughput checks.",
    )
    parser.add_argument(
        "--external-max-rows-per-source",
        type=int,
        default=50000,
        help="Mandatory pre-step: external dataset rows to materialize per source.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    ensure_external_dataset_prestep(
        repo_root=base_dir,
        max_rows_per_source=int(args.external_max_rows_per_source),
    )
    rows = _load_rows(Path(args.jsonl))
    gates = yaml.safe_load(Path(args.gates_config).read_text(encoding="utf-8")) or {}
    quality = gates.get("censorship_quality_gates", {})
    latency_cfg = gates.get("censorship_latency_gates", {})
    latency_mode = args.latency_mode
    if latency_mode == "auto":
        latency_mode = "replay" if "replay" in Path(args.jsonl).stem.lower() else "live"

    total = len(rows)
    review_rate = _ratio(sum(1 for r in rows if _decision(r) == "review"), total)
    reason_coverage = _ratio(sum(1 for r in rows if _reason_codes(r)), total)
    hard_block_rate = _ratio(sum(1 for r in rows if _decision(r) == "hard_block"), total)
    review_rate_max = float(
        quality.get(
            "review_rate_replay_max" if latency_mode == "replay" else "review_rate_max",
            quality.get("review_rate_max", 1.0),
        )
    )
    summary = {
        "rows": total,
        "review_rate": review_rate,
        "reason_coverage": reason_coverage,
        "hard_block_rate": hard_block_rate,
        "review_rate_max": review_rate_max,
        "reason_coverage_min": float(quality.get("reason_coverage_min", 0.0)),
        "latency_mode": latency_mode,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = []
    if review_rate > summary["review_rate_max"]:
        failed.append("review_rate")
    if reason_coverage < summary["reason_coverage_min"]:
        failed.append("reason_coverage")

    if bool(latency_cfg.get("enabled", False)) and args.metrics_json:
        metrics = _load_json(Path(args.metrics_json))
        current = {
            "p95_latency_ms": float(metrics.get("p95_latency_ms", 0.0)),
            "throughput_items_per_second": float(metrics.get("throughput_items_per_second", 0.0)),
            "l3_used_share": float(metrics.get("l3_used_share", 0.0)),
        }
        baseline, bootstrapped = _bootstrap_or_load_latency_baseline(
            Path(args.latency_baseline_json),
            current,
        )
        p95_delta_max = float(
            latency_cfg.get(
                "p95_with_l3_delta_max_ms" if current["l3_used_share"] > 0 else "p95_without_l3_delta_max_ms",
                40.0,
            )
        )
        p95_ratio_max = float(latency_cfg.get("p95_ratio_vs_baseline_max", 1.10))
        throughput_ratio_min = float(latency_cfg.get("throughput_min_ratio_vs_baseline", 0.8))
        baseline_p95 = max(float(baseline.get("p95_latency_ms", 0.0)), 1e-6)
        baseline_thr = max(float(baseline.get("throughput_items_per_second", 0.0)), 1e-6)
        p95_delta = current["p95_latency_ms"] - baseline_p95
        p95_ratio = current["p95_latency_ms"] / baseline_p95
        throughput_ratio = current["throughput_items_per_second"] / baseline_thr
        latency_summary = {
            "latency_baseline_bootstrapped": bootstrapped,
            "current_p95_latency_ms": current["p95_latency_ms"],
            "baseline_p95_latency_ms": baseline_p95,
            "p95_delta_ms": p95_delta,
            "p95_delta_max_ms": p95_delta_max,
            "p95_ratio": p95_ratio,
            "p95_ratio_max": p95_ratio_max,
            "current_throughput_items_per_second": current["throughput_items_per_second"],
            "baseline_throughput_items_per_second": baseline_thr,
            "throughput_ratio": throughput_ratio,
            "throughput_ratio_min": throughput_ratio_min,
        }
        print(json.dumps(latency_summary, ensure_ascii=False, indent=2))
        if p95_delta > p95_delta_max:
            failed.append("p95_delta_vs_baseline")
        if latency_mode == "live":
            if p95_ratio > p95_ratio_max:
                failed.append("p95_ratio_vs_baseline")
            if throughput_ratio < throughput_ratio_min:
                failed.append("throughput_ratio_vs_baseline")
        else:
            print("INFO replay latency mode: skipped relative p95/throughput gates")
    if failed:
        print(f"FAILED gates={','.join(failed)}")
        return 2
    print("OK gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


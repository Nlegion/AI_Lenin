"""Auto-rollback censorship mode to shadow when degradation gates fail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--release-gates", default="config/release_gates.yaml")
    parser.add_argument("--safety-config", default="config/safety_gate_config.yaml")
    args = parser.parse_args()

    gates = _load_yaml(Path(args.release_gates)).get("censorship_latency_gates", {})
    current = _load_json(Path(args.metrics_json))
    baseline = _load_json(Path(args.baseline_json))
    p95_current = float(current.get("p95_latency_ms", 0.0))
    p95_base = max(float(baseline.get("p95_latency_ms", 0.0)), 1e-6)
    thr_current = float(current.get("throughput_items_per_second", 0.0))
    thr_base = max(float(baseline.get("throughput_items_per_second", 0.0)), 1e-6)

    p95_ratio = p95_current / p95_base
    thr_ratio = thr_current / thr_base
    p95_ratio_max = float(gates.get("p95_ratio_vs_baseline_max", 1.10))
    thr_ratio_min = float(gates.get("throughput_min_ratio_vs_baseline", 0.8))
    degraded = p95_ratio > p95_ratio_max or thr_ratio < thr_ratio_min

    safety_path = Path(args.safety_config)
    safety = _load_yaml(safety_path)
    flags = safety.setdefault("safety_gate", {}).setdefault("flags", {})
    if degraded:
        flags["shadow_mode"] = True
        flags["enforce_mode"] = "old"
        safety_path.write_text(
            yaml.safe_dump(safety, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    result = {
        "degraded": degraded,
        "p95_ratio": p95_ratio,
        "p95_ratio_max": p95_ratio_max,
        "throughput_ratio": thr_ratio,
        "throughput_ratio_min": thr_ratio_min,
        "shadow_mode": flags.get("shadow_mode"),
        "enforce_mode": flags.get("enforce_mode"),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


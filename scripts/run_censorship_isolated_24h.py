"""Run isolated censorship experiment without RAG/generation/publish."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.core.safety.censor_hashing import compute_content_hash
from src.core.safety.pre_rag_censor import CensorRuntimeConfig, PreRagCensor
from src.core.safety.pre_rag_censor_types import CensorInput
from src.core.safety.safety_gate import SafetyGate
from src.core.safety.news_guard import NewsGuard
from src.core.settings.censorship_runtime_config import (
    default_censorship_runtime_config_path,
    load_censorship_runtime_config,
)
from src.modules.news_system.fetcher import NewsFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("censorship_24h")


def _load_cfg(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("censorship_experiment", payload)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "timestamp_utc",
        "news_id",
        "source",
        "language",
        "title",
        "body_hash",
        "decision",
        "category",
        "risk_tier",
        "reason_codes",
        "l2_similarity",
        "l3_used",
        "latency_ms",
        "config_version_hash",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def _safe_git_head(*, base_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return result.stdout.strip() or "unknown"


def _validate_control_rows(*, rows: list[dict[str, Any]], min_rows: int) -> None:
    if len(rows) < min_rows:
        raise ValueError(f"control set too small: len={len(rows)} required={min_rows}")
    required = ("id", "title", "content")
    for idx, row in enumerate(rows, start=1):
        for field in required:
            value = str(row.get(field) or "").strip()
            if not value:
                raise ValueError(f"control row {idx} missing required field '{field}'")


def _build_payload(
    *,
    item: dict[str, Any],
    result: Any,
    source_trust_tier: str,
    dataset: str | None = None,
) -> dict[str, Any]:
    audit = result.audit if isinstance(result.audit, dict) else {}
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "news_id": str(item.get("id")),
        "source": str(item.get("source") or "unknown"),
        "source_trust_tier": source_trust_tier,
        "language": str(item.get("language") or "ru"),
        "title": str(item.get("title") or ""),
        "body_hash": compute_content_hash(
            title=str(item.get("title") or ""),
            body=str(item.get("content") or ""),
            url=str(item.get("url") or ""),
        )[0],
        "decision": result.decision,
        "category": result.category,
        "risk_tier": result.risk_tier,
        "reason_codes": result.reason_codes,
        "l1_decision": audit.get("l1_decision"),
        "l2_similarity": result.confidence.get("l2_similarity"),
        "l3_used": bool(result.confidence.get("l3_used")),
        "normalization_flags": audit.get("normalization", {}),
        "latency_ms": float(audit.get("latency_ms", 0.0)),
        "config_version_hash": (
            audit.get("config_version_hash")
            or audit.get("shadow", {}).get("config_version_hash")
            or audit.get("runtime_config_hash")
            or ""
        ),
        "dataset": dataset or "live",
    }
    return payload


def _append_sidecar_if_needed(
    *,
    sidecar_path: Path | None,
    item: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    if sidecar_path is None:
        return
    if payload.get("decision") not in {"allow", "review"}:
        return
    sidecar = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "news_id": payload.get("news_id"),
        "source": payload.get("source"),
        "language": payload.get("language"),
        "title": payload.get("title"),
        "content": str(item.get("content") or ""),
        "decision": payload.get("decision"),
        "category": payload.get("category"),
        "risk_tier": payload.get("risk_tier"),
        "l1_decision": payload.get("l1_decision"),
        "reason_codes": payload.get("reason_codes"),
        "l2_similarity": payload.get("l2_similarity"),
        "config_version_hash": payload.get("config_version_hash"),
        "dataset": payload.get("dataset", "live"),
    }
    _append_jsonl(sidecar_path, sidecar)


def _summary(rows: list[dict[str, Any]], *, elapsed_seconds: float) -> dict[str, Any]:
    def _rates(items: list[dict[str, Any]]) -> dict[str, float]:
        total = max(len(items), 1)
        hard = sum(1 for r in items if r.get("decision") == "hard_block")
        review = sum(1 for r in items if r.get("decision") == "review")
        skip = sum(1 for r in items if r.get("decision") == "skip")
        allow = sum(1 for r in items if r.get("decision") == "allow")
        return {
            "hard_block_rate": hard / total,
            "review_rate": review / total,
            "skip_rate": skip / total,
            "allow_rate": allow / total,
            "reason_coverage": sum(1 for r in items if r.get("reason_codes")) / total,
        }

    def _latency(items: list[dict[str, Any]]) -> dict[str, float]:
        latencies = [float(r.get("latency_ms") or 0.0) for r in items]
        latencies.sort()
        p50 = latencies[int(0.5 * (len(latencies) - 1))] if latencies else 0.0
        p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
        return {"p50_latency_ms": p50, "p95_latency_ms": p95}

    total = max(len(rows), 1)
    core_rows = [
        r
        for r in rows
        if "l0_filtered" not in (r.get("reason_codes") or [])
        and not bool((r.get("normalization_flags") or {}).get("duplicate_hit"))
    ]
    summary: dict[str, Any] = {
        "n": len(rows),
        **_rates(rows),
        **_latency(rows),
        "elapsed_seconds": max(float(elapsed_seconds), 0.001),
        "throughput_items_per_second": len(rows) / max(float(elapsed_seconds), 0.001),
        "n_policy_core": len(core_rows),
        "policy_core_share": len(core_rows) / total,
        "policy_core_rates": _rates(core_rows) if core_rows else {},
        "policy_core_latency": _latency(core_rows) if core_rows else {},
        "l0_filtered_share": sum(1 for r in rows if "l0_filtered" in (r.get("reason_codes") or [])) / total,
    }
    l3_rows = [r for r in rows if bool(r.get("l3_used"))]
    no_l3_rows = [r for r in rows if not bool(r.get("l3_used"))]
    summary["l3_used_share"] = len(l3_rows) / total
    summary["latency_with_l3"] = _latency(l3_rows) if l3_rows else {}
    summary["latency_without_l3"] = _latency(no_l3_rows) if no_l3_rows else {}
    return summary


def _run_once(
    *,
    censor: PreRagCensor,
    fetcher: NewsFetcher,
    jsonl_path: Path,
    sidecar_path: Path | None,
) -> int:
    items = fetcher.fetch_all()
    for item in items:
        result = asyncio_run(
            censor.evaluate(
                CensorInput(
                    news_id=str(item.get("id")),
                    title=str(item.get("title") or ""),
                    body=str(item.get("content") or ""),
                    source=str(item.get("source") or "unknown"),
                    metadata={"url": str(item.get("url") or "")},
                )
            )
        )
        payload = _build_payload(
            item=item,
            result=result,
            source_trust_tier="trusted"
            if str(item.get("source") or "").upper() == "TASS"
            else "unknown",
            dataset="live",
        )
        _append_jsonl(jsonl_path, payload)
        _append_sidecar_if_needed(sidecar_path=sidecar_path, item=item, payload=payload)
    return len(items)


def _run_control_once(
    *,
    censor: PreRagCensor,
    control_rows: list[dict[str, Any]],
    jsonl_path: Path,
    take: int,
    cursor: int,
    sidecar_path: Path | None,
) -> tuple[int, int]:
    if not control_rows or take <= 0 or cursor >= len(control_rows):
        return 0, cursor
    end = min(cursor + max(take, 0), len(control_rows))
    rows = control_rows[cursor:end]
    count = 0
    for item in rows:
        result = asyncio_run(
            censor.evaluate(
                CensorInput(
                    news_id=str(item.get("id") or f"ctrl-{count}"),
                    title=str(item.get("title") or ""),
                    body=str(item.get("content") or ""),
                    source=str(item.get("source") or "CONTROL"),
                    metadata={"url": str(item.get("url") or "")},
                )
            )
        )
        payload = _build_payload(
            item=item,
            result=result,
            source_trust_tier="control",
            dataset="control",
        )
        _append_jsonl(jsonl_path, payload)
        _append_sidecar_if_needed(sidecar_path=sidecar_path, item=item, payload=payload)
        count += 1
    return count, end


def asyncio_run(awaitable):
    import asyncio

    return asyncio.run(awaitable)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/censorship_experiment.yaml")
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete prior output files before run",
    )
    parser.add_argument(
        "--control-jsonl",
        default=None,
        help="Optional labeled/control JSONL with fields title/content/source/language",
    )
    parser.add_argument(
        "--control-batch-size",
        type=int,
        default=0,
        help="How many control items to process per loop",
    )
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    duration_hours = float(args.duration_hours or cfg.get("duration_hours", 24))
    poll_seconds = int(cfg.get("poll_seconds", 300))
    out_dir = Path(cfg.get("output_dir", ".cursor/artifacts/quality"))
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).resolve().parents[1]
    gate = SafetyGate.from_base_dir(base_dir)
    guard = NewsGuard.from_file(base_dir / "config" / "news_guard.yaml")
    cfg_path = default_censorship_runtime_config_path(base_dir)
    runtime = load_censorship_runtime_config(cfg_path)
    overrides = cfg.get("feature_flags", {})
    if overrides:
        runtime = CensorRuntimeConfig(**{**runtime.__dict__, **overrides})
    censor = PreRagCensor(
        safety_gate=gate,
        news_guard=guard,
        config=runtime,
        config_path=str(cfg_path),
    )
    fetcher = NewsFetcher()

    random.seed(int(cfg.get("seed", 42)))
    control_rows: list[dict[str, Any]] = []
    control_path_value = args.control_jsonl or cfg.get("control_jsonl")
    if control_path_value:
        cp = Path(str(control_path_value))
        if cp.is_file():
            control_rows = _read_jsonl(cp)
            random.shuffle(control_rows)
            logger.info("loaded_control_rows=%s path=%s", len(control_rows), cp)
    control_batch_size = int(args.control_batch_size or cfg.get("control_batch_size", 0))
    if control_batch_size > 0 and not control_rows:
        logger.error("control_batch_size=%s but control set is empty/missing", control_batch_size)
        return 1
    if control_batch_size > 0:
        try:
            _validate_control_rows(rows=control_rows, min_rows=control_batch_size)
        except ValueError as error:
            logger.error("invalid control set: %s", error)
            return 1
    started = time.time()
    deadline = started + duration_hours * 3600
    jsonl_path = out_dir / str(cfg.get("jsonl_filename", "censorship_24h_latest.jsonl"))
    csv_path = out_dir / str(cfg.get("csv_filename", "censorship_24h_latest.csv"))
    metrics_path = out_dir / str(cfg.get("metrics_filename", "censorship_24h_latest.metrics.json"))
    notes_path = out_dir / str(cfg.get("notes_filename", "censorship_24h_latest.notes.md"))
    persist_allow_bodies = bool(cfg.get("persist_allow_bodies", False))
    sidecar_path = (
        out_dir / str(cfg.get("allow_bodies_filename", "censorship_allow_bodies.jsonl"))
        if persist_allow_bodies
        else None
    )
    if args.fresh:
        for path in (jsonl_path, csv_path, metrics_path, notes_path, sidecar_path):
            if path is None:
                continue
            if path.exists():
                path.unlink()

    total = 0
    control_cursor = 0
    control_exhausted = False
    last_config_hash = ""
    while time.time() < deadline:
        processed = _run_once(
            censor=censor,
            fetcher=fetcher,
            jsonl_path=jsonl_path,
            sidecar_path=sidecar_path,
        )
        control_processed, control_cursor = _run_control_once(
            censor=censor,
            control_rows=control_rows,
            jsonl_path=jsonl_path,
            take=control_batch_size,
            cursor=control_cursor,
            sidecar_path=sidecar_path,
        )
        total += processed + control_processed
        if control_processed == 0 and control_batch_size > 0 and control_cursor >= len(control_rows):
            control_exhausted = True
        logger.info(
            "processed batch news=%s control=%s total=%s control_cursor=%s/%s",
            processed,
            control_processed,
            total,
            control_cursor,
            len(control_rows),
        )
        remaining = max(deadline - time.time(), 0.0)
        if remaining <= 0:
            break
        time.sleep(min(poll_seconds, remaining))

    rows = _read_jsonl(jsonl_path)
    _export_csv(csv_path, rows)
    elapsed_seconds = max(time.time() - started, 0.001)
    summary = _summary(rows, elapsed_seconds=elapsed_seconds)
    if rows:
        last_config_hash = str(rows[-1].get("config_version_hash") or "")
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    git_head = _safe_git_head(base_dir=base_dir)
    notes_path.write_text(
        "\n".join(
            [
                "# Censorship Isolated Run",
                f"- started_utc: {datetime.fromtimestamp(started, tz=timezone.utc).isoformat()}",
                f"- duration_hours: {duration_hours}",
                f"- rows: {len(rows)}",
                f"- poll_seconds: {poll_seconds}",
                f"- jsonl: {jsonl_path.name}",
                f"- csv: {csv_path.name}",
                f"- metrics: {metrics_path.name}",
                f"- control_path: {control_path_value or ''}",
                f"- control_rows_total: {len(control_rows)}",
                f"- control_batch_size: {control_batch_size}",
                f"- control_cursor_final: {control_cursor}",
                f"- control_exhausted: {control_exhausted}",
                f"- sidecar: {(sidecar_path.name if sidecar_path else 'disabled')}",
                f"- config_version_hash_last: {last_config_hash}",
                f"- git_head: {git_head}",
                f"- python_version: {sys.version.split()[0]}",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("done rows=%s p95=%.2fms", len(rows), float(summary.get("p95_latency_ms", 0.0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

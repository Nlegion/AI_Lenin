#!/usr/bin/env python3
"""Watch live Qdrant ingest progress from checkpoint file (for already-running jobs)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import time

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_offset(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch ingest checkpoint progress.")
    parser.add_argument(
        "--checkpoint",
        default=".cursor/artifacts/qdrant/checkpoints/ingestion_giga_v1.offset",
    )
    parser.add_argument("--total", type=int, default=51914, help="Expected total rows.")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll seconds.")
    args = parser.parse_args()

    checkpoint = (REPO_ROOT / args.checkpoint).resolve()
    total = max(args.total, 1)
    prev = _read_offset(checkpoint)
    base = prev
    started = time.perf_counter()
    print(
        f"[watch] checkpoint={checkpoint} total={total} interval={args.interval}s",
        flush=True,
    )
    print(
        f"[watch] {datetime.now(UTC).strftime('%H:%M:%SZ')} "
        f"offset={prev}/{total} ({100.0 * prev / total:5.1f}%)",
        flush=True,
    )

    try:
        while True:
            time.sleep(args.interval)
            current = _read_offset(checkpoint)
            elapsed = time.perf_counter() - started
            delta = current - prev
            window_rate = (max(delta, 0) / args.interval) if args.interval > 0 else 0.0
            overall_rate = ((current - base) / elapsed) if elapsed > 0 else 0.0
            rate = window_rate if window_rate > 0 else overall_rate
            remaining = max(total - current, 0)
            eta_txt = f"{(remaining / rate / 60.0):5.1f}m" if rate > 0 else "  n/a"
            moved = "up" if current > prev else ("same" if current == prev else "down")
            print(
                f"[watch] {datetime.now(UTC).strftime('%H:%M:%SZ')} "
                f"{moved} {current}/{total} ({100.0 * current / total:5.1f}%) "
                f"d={delta:+d}/{args.interval:.0f}s "
                f"rate={rate:5.2f} rows/s eta={eta_txt}",
                flush=True,
            )
            prev = current
            if current >= total:
                print("[watch] checkpoint reached total; exiting", flush=True)
                return 0
    except KeyboardInterrupt:
        print("\n[watch] interrupted", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

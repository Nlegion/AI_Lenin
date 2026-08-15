"""Build control JSONL for isolated censorship experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _to_control_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    text = str(row.get("text") or "").strip()
    title = text.split(". ")[0][:160] if text else "control item"
    return {
        "id": f"ctrl-{idx}",
        "title": title,
        "content": text,
        "source": f"CONTROL::{row.get('source') or 'external'}",
        "language": "ru",
        "expected_category": str(row.get("category") or "UNKNOWN"),
        "expected_label": str(row.get("label") or "UNKNOWN"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl", default="data/external_datasets/external_unified.jsonl"
    )
    parser.add_argument(
        "--output-jsonl",
        default=".cursor/artifacts/quality/censorship_control_set_latest.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total", type=int, default=1500)
    parser.add_argument("--min-per-source", type=int, default=250)
    args = parser.parse_args()

    rnd = random.Random(args.seed)
    rows = _read_jsonl(Path(args.input_jsonl))
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        src = str(row.get("source") or "external")
        by_source.setdefault(src, []).append(row)

    sampled: list[dict[str, Any]] = []
    for src_rows in by_source.values():
        copy = list(src_rows)
        rnd.shuffle(copy)
        sampled.extend(copy[: args.min_per_source])

    remaining = [row for row in rows if row not in sampled]
    rnd.shuffle(remaining)
    headroom = max(args.max_total - len(sampled), 0)
    sampled.extend(remaining[:headroom])

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(sampled, start=1):
            handle.write(
                json.dumps(_to_control_row(row, idx), ensure_ascii=False) + "\n"
            )
    print(f"control_rows={len(sampled)} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

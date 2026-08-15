"""Build deterministic stratified sample from censorship JSONL."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    decision = str(row.get("decision") or "unknown")
    category = str(row.get("category") or "none")
    source = str(row.get("source") or "unknown")
    return decision, category, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to censorship JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-category", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=2000)
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=["NON_TOPICAL"],
        help="Category to deprioritize in base stratified pool (repeatable)",
    )
    parser.add_argument("--max-non-topical", type=int, default=200)
    args = parser.parse_args()

    rnd = random.Random(args.seed)
    rows = _load_rows(Path(args.input))
    excluded = set(args.exclude_category or [])
    topical_rows = [r for r in rows if str(r.get("category")) not in excluded]
    non_topical_rows = [r for r in rows if str(r.get("category")) in excluded]
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in topical_rows:
        by_group[_group_key(row)].append(row)

    sampled: list[dict[str, Any]] = []
    category_groups: dict[
        str, list[tuple[tuple[str, str, str], list[dict[str, Any]]]]
    ] = defaultdict(list)
    for key, items in by_group.items():
        category_groups[key[1]].append((key, items))

    for category, groups in category_groups.items():
        bucket: list[dict[str, Any]] = []
        for _key, items in groups:
            copy = list(items)
            rnd.shuffle(copy)
            bucket.extend(copy)
        rnd.shuffle(bucket)
        take = min(len(bucket), args.min_per_category)
        sampled.extend(bucket[:take])

    remaining = [r for r in topical_rows if r not in sampled]
    rnd.shuffle(remaining)
    headroom = max(args.max_total - len(sampled), 0)
    sampled.extend(remaining[:headroom])
    if non_topical_rows and len(sampled) < args.max_total:
        rnd.shuffle(non_topical_rows)
        take_non_topical = min(
            len(non_topical_rows),
            args.max_non_topical,
            args.max_total - len(sampled),
        )
        sampled.extend(non_topical_rows[:take_non_topical])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in sampled:
            row = dict(row)
            row["sample_seed"] = args.seed
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"sampled={len(sampled)} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

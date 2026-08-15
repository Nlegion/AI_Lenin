"""Build quality QA input JSONL from censorship allow/review sidecar."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _question(title: str) -> str:
    return f"Прокомментируйте с позиций Ленина: {title}"


def _category(row: dict[str, Any]) -> str:
    return str(row.get("category") or "none").strip() or "none"


def _validate_row(row: dict[str, Any]) -> bool:
    required = ("news_id", "title", "content", "decision")
    for field in required:
        if not str(row.get(field) or "").strip():
            return False
    return str(row.get("decision") or "") in {"allow", "review"}


def _dedupe_allow_over_review(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        news_id = str(row.get("news_id") or "").strip()
        if not news_id:
            continue
        existing = by_id.get(news_id)
        if existing is None:
            by_id[news_id] = row
            continue
        existing_decision = str(existing.get("decision") or "")
        current_decision = str(row.get("decision") or "")
        if existing_decision == "review" and current_decision == "allow":
            by_id[news_id] = row
    return list(by_id.values())


def _round_robin_pick(
    *,
    rows: list[dict[str, Any]],
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_category(row)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered_keys = sorted(buckets.keys())
    out: list[dict[str, Any]] = []
    while len(out) < limit and ordered_keys:
        next_keys: list[str] = []
        for key in ordered_keys:
            items = buckets[key]
            if items:
                out.append(items.pop())
                if len(out) >= limit:
                    break
            if items:
                next_keys.append(key)
        ordered_keys = next_keys
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodies", required=True, help="Allow/review sidecar JSONL")
    parser.add_argument("--output", required=True, help="Output quality QA JSONL")
    parser.add_argument("--limit-per-decision", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.bodies))
    valid_rows = [row for row in rows if _validate_row(row)]
    deduped_rows = _dedupe_allow_over_review(valid_rows)
    by_decision: dict[str, list[dict[str, Any]]] = {"allow": [], "review": []}
    for row in deduped_rows:
        by_decision[str(row["decision"])].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for decision in ("allow", "review"):
        selected.extend(
            _round_robin_pick(
                rows=by_decision[decision],
                limit=max(args.limit_per_decision, 0),
                rng=rng,
            )
        )

    if not selected:
        if args.allow_empty:
            print("gen_skipped=true reason=no_allow_or_review_rows")
            return 0
        print("gen_skipped=true reason=no_allow_or_review_rows")
        return 10

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in selected:
            title = str(row.get("title") or "").strip()
            content = str(row.get("content") or "").strip()
            if not title or not content:
                continue
            qa_row = {
                "id": str(row.get("news_id")),
                "title": title,
                "content": content,
                "question": _question(title),
                "topic": str(row.get("decision")),
                "source": str(row.get("source") or ""),
            }
            handle.write(json.dumps(qa_row, ensure_ascii=False) + "\n")

    print(
        "qa_rows=%s allow_rows=%s review_rows=%s output=%s"
        % (
            len(selected),
            len([r for r in selected if r.get("decision") == "allow"]),
            len([r for r in selected if r.get("decision") == "review"]),
            out,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

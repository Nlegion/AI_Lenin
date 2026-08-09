"""Apply manual review decisions and emit reingestion payload."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="Path to review_queue_*.jsonl")
    parser.add_argument(
        "--decisions",
        required=True,
        help="JSONL with fields: review_id, final_decision(review_approved|review_rejected)",
    )
    parser.add_argument("--out", required=True, help="Output JSONL for reingestion jobs")
    args = parser.parse_args()

    queue_rows = _read_jsonl(Path(args.queue))
    decision_rows = _read_jsonl(Path(args.decisions))
    decision_by_id = {r["review_id"]: r for r in decision_rows if r.get("review_id")}

    updated_queue: list[dict] = []
    reingest_jobs: list[dict] = []
    for row in queue_rows:
        review_id = row.get("review_id")
        manual = decision_by_id.get(review_id)
        if manual is None:
            updated_queue.append(row)
            continue
        final_decision = manual.get("final_decision", "review_rejected")
        resolved = {
            **row,
            "status": final_decision,
            "final_decision": "allow" if final_decision == "review_approved" else "hard_block",
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": "manual_review",
        }
        updated_queue.append(resolved)
        if final_decision == "review_approved":
            reingest_jobs.append(
                {
                    "news_id": row.get("news_id"),
                    "title": row.get("title"),
                    "body": row.get("body"),
                    "source": row.get("source"),
                    "manual_override": True,
                    "origin_review_id": review_id,
                }
            )

    _write_jsonl(Path(args.queue), updated_queue)
    _write_jsonl(Path(args.out), reingest_jobs)
    print(f"updated_queue={len(updated_queue)} reingest_jobs={len(reingest_jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

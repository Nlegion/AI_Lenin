#!/usr/bin/env python3
"""Collect anti-cliché label-batch candidates from gold JSONL + optional dry-run hints."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a weekly human-label batch toward the ≥50 unique-pair bar."
    )
    parser.add_argument("--cases", default="data/eval/anti_cliche_cases.jsonl")
    parser.add_argument(
        "--out-dir",
        default=".cursor/artifacts/human_eval",
        help="Output directory for batch stub.",
    )
    args = parser.parse_args()

    cases_path = (REPO_ROOT / args.cases).resolve()
    rows: list[dict] = []
    if cases_path.is_file():
        with cases_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = out_dir / f"label_batch_{stamp}.jsonl"
    summary_path = out_dir / f"label_batch_{stamp}_summary.md"

    unique: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("id", "")), str(row.get("analysis", ""))[:200])
        unique[key] = {
            "id": row.get("id"),
            "query": row.get("news_title") or row.get("id"),
            "answer": row.get("analysis"),
            "machine_expect_codes": row.get("expect_codes", []),
            "human_label": None,
            "notes": "",
        }

    with batch_path.open("w", encoding="utf-8") as handle:
        for item in unique.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_path.write_text(
        "\n".join(
            [
                f"# Label batch {stamp}",
                "",
                f"- Unique candidates: `{len(unique)}`",
                "- Prefer adding recent dry-run `(query, answer)` pairs before labeling.",
                "- Target mix ~50% warn / ~50% pass; dedupe `(query, answer)`.",
                "- See `docs/human_eval_checklist.md` for the weekly loop and H1-d bar.",
                f"- Progress toward 50: `{len(unique)}` seeded from gold (expand weekly).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {batch_path} candidates={len(unique)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bulk rebuild ultimate_cleaned_ontology from intellectual sources."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.preprocessing.cleaning_config import load_cleaning_config  # noqa: E402
from src.core.preprocessing.text_cleaner import clean_document  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk repair ultimate_cleaned_ontology corpus.")
    parser.add_argument("--src-root", default="data/books/intellectual")
    parser.add_argument("--dst-root", default="data/books/ultimate_cleaned_ontology")
    parser.add_argument("--config", default="config/cleaning_rules.yaml")
    parser.add_argument(
        "--stats-json",
        default=".cursor/artifacts/cleaning/bulk_repair_ultimate_cleaned_stats.json",
    )
    parser.add_argument(
        "--summary-md",
        default=".cursor/artifacts/cleaning/bulk_repair_ultimate_cleaned_summary.md",
    )
    args = parser.parse_args()

    src_root = (REPO_ROOT / args.src_root).resolve()
    dst_root = (REPO_ROOT / args.dst_root).resolve()
    config = load_cleaning_config(path=(REPO_ROOT / args.config).resolve())

    source_files = sorted(src_root.rglob("*.txt"))
    stats: list[dict[str, str | int | bool]] = []
    by_author = defaultdict(lambda: {"files": 0, "lines_min": 10**9, "lines_max": 0})

    for src_path in source_files:
        relative = src_path.relative_to(src_root)
        author = relative.parts[0] if relative.parts else "unknown"
        work = src_path.stem
        original = src_path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_document(text=original, config=config)
        output_text = f"АВТОР: {author}\nРАБОТА: {work}\n\n{cleaned}\n"

        target_path = dst_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(output_text, encoding="utf-8")

        lines = len(output_text.splitlines())
        by_author[author]["files"] += 1
        by_author[author]["lines_min"] = min(by_author[author]["lines_min"], lines)
        by_author[author]["lines_max"] = max(by_author[author]["lines_max"], lines)
        stats.append(
            {
                "rel_path": relative.as_posix(),
                "author": author,
                "work": work,
                "original_chars": len(original),
                "cleaned_chars": len(cleaned),
                "output_lines": lines,
                "has_print_meta": "Подписано к печати" in output_text,
            }
        )

    stats_payload = {
        "files_total": len(stats),
        "authors_total": len(by_author),
        "authors": dict(by_author),
        "stats": stats,
    }

    stats_path = (REPO_ROOT / args.stats_json).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = (REPO_ROOT / args.summary_md).resolve()
    summary_lines = [
        "# Bulk Repair Ultimate Cleaned Ontology",
        "",
        f"- Files total: `{len(stats)}`",
        f"- Authors total: `{len(by_author)}`",
        "",
        "| Author | Files | Min lines | Max lines |",
        "|---|---:|---:|---:|",
    ]
    for author in sorted(by_author):
        row = by_author[author]
        summary_lines.append(
            f"| {author} | {row['files']} | {row['lines_min']} | {row['lines_max']} |"
        )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"files_total {len(stats)}")
    print(f"authors_total {len(by_author)}")
    print(f"with_print_meta {sum(1 for item in stats if item['has_print_meta'])}")
    print(f"min_lines {min(item['output_lines'] for item in stats)}")
    print(f"max_lines {max(item['output_lines'] for item in stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

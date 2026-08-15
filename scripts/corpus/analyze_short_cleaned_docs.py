#!/usr/bin/env python3
"""Inspect short cleaned ontology docs for corruption indicators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze short cleaned documents quality."
    )
    parser.add_argument(
        "--stats-json",
        default=".cursor/artifacts/cleaning/bulk_repair_ultimate_cleaned_stats.json",
    )
    parser.add_argument("--root", default="data/books/ultimate_cleaned_ontology")
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument(
        "--out-md", default=".cursor/artifacts/cleaning/short_docs_qc.md"
    )
    parser.add_argument(
        "--out-json", default=".cursor/artifacts/cleaning/short_docs_qc.json"
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    stats_path = (repo / args.stats_json).resolve()
    root = (repo / args.root).resolve()

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    short_docs = [
        row for row in payload["stats"] if int(row["output_lines"]) < args.max_lines
    ]

    findings: list[dict[str, str | int | float]] = []
    for row in sorted(short_docs, key=lambda item: int(item["output_lines"])):
        path = root / Path(str(row["rel_path"]))
        text = path.read_text(encoding="utf-8", errors="replace")
        body = "\n".join(text.splitlines()[3:])
        words = re.findall(r"[А-Яа-яЁёA-Za-z]{2,}", body)
        alpha_ratio = sum(ch.isalpha() for ch in body) / max(1, len(body))
        bad_chars = len(re.findall(r"[�□◆◊]", body))
        max_word_len = max((len(word) for word in words), default=0)
        findings.append(
            {
                "rel_path": str(row["rel_path"]),
                "lines": int(row["output_lines"]),
                "alpha_ratio": round(alpha_ratio, 4),
                "bad_chars": bad_chars,
                "max_word_len": max_word_len,
            }
        )

    out_json_path = (repo / args.out_json).resolve()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(
        json.dumps(
            {"short_docs_total": len(findings), "findings": findings},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Short Docs QC",
        "",
        f"- Max lines threshold: `{args.max_lines}`",
        f"- Short docs total: `{len(findings)}`",
        "",
        "| Path | Lines | Alpha ratio | Bad chars | Max word len |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in findings:
        lines.append(
            f"| `{item['rel_path']}` | {item['lines']} | {item['alpha_ratio']} | {item['bad_chars']} | {item['max_word_len']} |"
        )
    out_md_path = (repo / args.out_md).resolve()
    out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"short_docs_total {len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

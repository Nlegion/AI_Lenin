#!/usr/bin/env python3
"""Build machine-readable source registry for data/books corpus."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.utils.source_registry import (
    build_registry_summary,
    build_source_registry,
    export_source_registry_tsv,
    load_source_registry_rules,
)


def _write_summary_markdown(summary: dict[str, int], output_path: Path, corpus_root: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Source Registry Summary",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Corpus root: `{corpus_root.as_posix()}`",
        f"- Total records: `{summary['total_records']}`",
        f"- core_self: `{summary['stance_core_self']}`",
        f"- influence_agree: `{summary['stance_influence_agree']}`",
        f"- influence_critical: `{summary['stance_influence_critical']}`",
        f"- contextual: `{summary['stance_contextual']}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source registry for corpus inventory.")
    parser.add_argument("--corpus-root", default="data/books", help="Corpus root directory path.")
    parser.add_argument(
        "--rules-config",
        default="config/source_registry_rules.yaml",
        help="Optional YAML config with source typing rules.",
    )
    parser.add_argument(
        "--output",
        default=".cursor/artifacts/registries/source_registry.tsv",
        help="Output TSV registry path.",
    )
    parser.add_argument(
        "--summary-output",
        default=".cursor/artifacts/registries/source_registry_summary.md",
        help="Output Markdown summary path.",
    )
    args = parser.parse_args()

    corpus_root = (REPO_ROOT / args.corpus_root).resolve()
    rules_path = (REPO_ROOT / args.rules_config).resolve()
    output_path = (REPO_ROOT / args.output).resolve()
    summary_output_path = (REPO_ROOT / args.summary_output).resolve()

    rules = load_source_registry_rules(config_path=rules_path)
    records = build_source_registry(corpus_root=corpus_root, rules=rules)
    export_source_registry_tsv(records=records, output_path=output_path)

    summary = build_registry_summary(records=records)
    _write_summary_markdown(summary=summary, output_path=summary_output_path, corpus_root=corpus_root)

    print(f"Registry written: {output_path}")
    print(f"Summary written: {summary_output_path}")
    print(f"Total records: {summary['total_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

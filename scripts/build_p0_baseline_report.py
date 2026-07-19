#!/usr/bin/env python3
"""Build P0 reproducibility and ontology integrity report."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path


def _read_manifest_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _extract_metric(markdown: str, label: str) -> str:
    for line in markdown.splitlines():
        if line.strip().startswith(f"- {label}:"):
            return line.split(":", maxsplit=1)[1].strip()
    return "N/A"


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    manifest_md = repo / ".cursor/artifacts/cleaning/corpus_repro_manifest_v1.md"
    manifest_tsv = repo / ".cursor/artifacts/cleaning/corpus_repro_manifest_v1.tsv"
    ontology_md = repo / ".cursor/artifacts/ontology/ontology_summary.md"
    short_qc_md = repo / ".cursor/artifacts/cleaning/short_docs_qc.md"
    output = repo / ".cursor/artifacts/20260718-2115-p0-baseline-integrity.md"

    manifest_md_text = manifest_md.read_text(encoding="utf-8")
    ontology_md_text = ontology_md.read_text(encoding="utf-8")
    short_qc_text = short_qc_md.read_text(encoding="utf-8")
    manifest_rows = _read_manifest_tsv(path=manifest_tsv)
    noisy_rows = [row for row in manifest_rows if int(row["bad_chars"]) > 0]

    lines = [
        "# P0 Baseline and Ontology Integrity Report",
        "",
        f"- Generated at (UTC): `{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        "",
        "## Reproducibility Baseline",
        f"- Files tracked: {_extract_metric(manifest_md_text, 'Files tracked')}",
        f"- Missing source matches: {_extract_metric(manifest_md_text, 'Missing source matches')}",
        f"- Short docs (<300 lines): {_extract_metric(manifest_md_text, 'Short docs (<300 lines)')}",
        f"- Low alpha docs (<0.70): {_extract_metric(manifest_md_text, 'Low alpha docs (<0.70)')}",
        f"- Docs with noisy chars: {_extract_metric(manifest_md_text, 'Docs with noisy chars')}",
        "- Manifest version: `1.0.0`",
        "- Artifact: `.cursor/artifacts/cleaning/corpus_repro_manifest_v1.tsv`",
        "",
        "## Ontology Integrity",
        f"- Tagged documents: {_extract_metric(ontology_md_text, 'Tagged documents')}",
        f"- Graph nodes: {_extract_metric(ontology_md_text, 'Graph nodes')}",
        f"- Graph edges: {_extract_metric(ontology_md_text, 'Graph edges')}",
        f"- Documents with contradiction hits: {_extract_metric(ontology_md_text, 'Documents with contradiction hits')}",
        f"- IAA: {_extract_metric(ontology_md_text, 'IAA (annotator_a vs annotator_b)')}",
        "",
        "## Short Documents QC",
        "- Source: `.cursor/artifacts/cleaning/short_docs_qc.md`",
        f"- Summary: `{_extract_metric(short_qc_text, 'Short docs total')}`",
        "",
        "## Noisy Character Findings",
        f"- Documents with noise markers: `{len(noisy_rows)}`",
    ]
    if noisy_rows:
        lines.append("")
        lines.append("| Path | bad_chars |")
        lines.append("|---|---:|")
        for row in noisy_rows:
            lines.append(f"| `{row['relative_path']}` | {row['bad_chars']} |")
    else:
        lines.append("- No noisy-char rows detected.")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

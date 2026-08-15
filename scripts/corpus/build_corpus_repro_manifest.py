#!/usr/bin/env python3
"""Build reproducibility manifest and QC diff for cleaned corpus."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_text_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.txt"))


def _source_from_cleaned_path(
    cleaned_root: Path, source_root: Path, cleaned_file: Path
) -> Path:
    relative = cleaned_file.relative_to(cleaned_root)
    return source_root / relative


def _qc_flags(cleaned_text: str) -> dict[str, float | int]:
    lines = cleaned_text.splitlines()
    body = "\n".join(lines[3:]) if len(lines) >= 3 else ""
    alpha_ratio = sum(char.isalpha() for char in body) / max(1, len(body))
    bad_chars = len(re.findall(r"[�□◆◊]", body))
    return {
        "lines": len(lines),
        "alpha_ratio": alpha_ratio,
        "bad_chars": bad_chars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build corpus reproducibility manifest."
    )
    parser.add_argument("--source-root", default="data/books/intellectual")
    parser.add_argument(
        "--cleaned-root", default="data/books/ultimate_cleaned_ontology"
    )
    parser.add_argument("--manifest-version", default="1.0.0")
    parser.add_argument(
        "--out-tsv", default=".cursor/artifacts/cleaning/corpus_repro_manifest_v1.tsv"
    )
    parser.add_argument(
        "--out-md", default=".cursor/artifacts/cleaning/corpus_repro_manifest_v1.md"
    )
    args = parser.parse_args()

    source_root = (REPO_ROOT / args.source_root).resolve()
    cleaned_root = (REPO_ROOT / args.cleaned_root).resolve()
    out_tsv = (REPO_ROOT / args.out_tsv).resolve()
    out_md = (REPO_ROOT / args.out_md).resolve()

    cleaned_files = _collect_text_files(root=cleaned_root)
    rows: list[dict[str, str | int | float]] = []
    mismatched_source = 0
    short_docs = 0
    low_alpha_docs = 0
    noisy_docs = 0

    for cleaned_file in cleaned_files:
        source_file = _source_from_cleaned_path(
            cleaned_root=cleaned_root,
            source_root=source_root,
            cleaned_file=cleaned_file,
        )
        source_exists = source_file.exists()
        if not source_exists:
            mismatched_source += 1
            source_chars = 0
            source_sha = ""
        else:
            source_text = source_file.read_text(encoding="utf-8", errors="replace")
            source_chars = len(source_text)
            source_sha = _sha256(path=source_file)

        cleaned_text = cleaned_file.read_text(encoding="utf-8", errors="replace")
        cleaned_chars = len(cleaned_text)
        cleaned_sha = _sha256(path=cleaned_file)
        flags = _qc_flags(cleaned_text=cleaned_text)

        if int(flags["lines"]) < 300:
            short_docs += 1
        if float(flags["alpha_ratio"]) < 0.70:
            low_alpha_docs += 1
        if int(flags["bad_chars"]) > 0:
            noisy_docs += 1

        rows.append(
            {
                "relative_path": cleaned_file.relative_to(cleaned_root).as_posix(),
                "source_exists": int(source_exists),
                "source_chars": source_chars,
                "source_sha256": source_sha,
                "cleaned_chars": cleaned_chars,
                "cleaned_sha256": cleaned_sha,
                "lines": int(flags["lines"]),
                "alpha_ratio": round(float(flags["alpha_ratio"]), 4),
                "bad_chars": int(flags["bad_chars"]),
            }
        )

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "relative_path",
        "source_exists",
        "source_chars",
        "source_sha256",
        "cleaned_chars",
        "cleaned_sha256",
        "lines",
        "alpha_ratio",
        "bad_chars",
    ]
    tsv_lines = ["\t".join(header)]
    for row in rows:
        tsv_lines.append("\t".join(str(row[column]) for column in header))
    out_tsv.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_lines = [
        "# Corpus Reproducibility Manifest",
        "",
        f"- Manifest version: `{args.manifest_version}`",
        f"- Generated at (UTC): `{generated_at}`",
        f"- Source root: `{source_root}`",
        f"- Cleaned root: `{cleaned_root}`",
        f"- Files tracked: `{len(rows)}`",
        f"- Missing source matches: `{mismatched_source}`",
        f"- Short docs (<300 lines): `{short_docs}`",
        f"- Low alpha docs (<0.70): `{low_alpha_docs}`",
        f"- Docs with noisy chars: `{noisy_docs}`",
        "",
        f"- TSV manifest: `{out_tsv.relative_to(REPO_ROOT).as_posix()}`",
    ]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"tracked_files {len(rows)}")
    print(f"missing_source {mismatched_source}")
    print(f"short_docs {short_docs}")
    print(f"low_alpha_docs {low_alpha_docs}")
    print(f"noisy_docs {noisy_docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

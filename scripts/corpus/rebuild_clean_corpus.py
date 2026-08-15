#!/usr/bin/env python3
"""Rebuild cleaned corpus from source registry with QA metrics."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.preprocessing.cleaning_config import load_cleaning_config  # noqa: E402
from src.core.preprocessing.cleaning_quality import semantic_damage_ratio  # noqa: E402
from src.core.preprocessing.text_cleaner import clean_document  # noqa: E402


def _read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_report_markdown(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cleaning Rebuild Summary",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Processed files: `{payload['processed_files']}`",
        f"- Written files: `{payload['written_files']}`",
        f"- Skipped files (too short after cleaning): `{payload['skipped_files']}`",
        f"- Mean size reduction: `{payload['mean_size_reduction_pct']:.2f}%`",
        f"- Validation sample size: `{payload['validation_sample_size']}`",
        f"- Mean semantic damage ratio: `{payload['mean_semantic_damage_ratio']:.4f}`",
        f"- Max semantic damage ratio: `{payload['max_semantic_damage_ratio_observed']:.4f}`",
        f"- Threshold (`<`): `{payload['max_semantic_damage_ratio_target']:.4f}`",
        f"- Threshold passed: `{'yes' if payload['threshold_passed'] else 'no'}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild cleaned corpus with QA.")
    parser.add_argument(
        "--registry", default=".cursor/artifacts/registries/source_registry.tsv"
    )
    parser.add_argument("--corpus-root", default="data/books")
    parser.add_argument("--config", default="config/cleaning_rules.yaml")
    parser.add_argument("--cleaned-root", default=".cursor/artifacts/cleaned_corpus")
    parser.add_argument(
        "--qa-json", default=".cursor/artifacts/cleaning/cleaning_qa.json"
    )
    parser.add_argument(
        "--summary-md", default=".cursor/artifacts/cleaning/cleaning_summary.md"
    )
    args = parser.parse_args()

    registry_path = (REPO_ROOT / args.registry).resolve()
    corpus_root = (REPO_ROOT / args.corpus_root).resolve()
    config_path = (REPO_ROOT / args.config).resolve()
    cleaned_root = (REPO_ROOT / args.cleaned_root).resolve()
    qa_json_path = (REPO_ROOT / args.qa_json).resolve()
    summary_md_path = (REPO_ROOT / args.summary_md).resolve()

    config = load_cleaning_config(path=config_path)
    rows = _read_registry(path=registry_path)
    if cleaned_root.exists():
        for file_path in sorted(cleaned_root.rglob("*"), reverse=True):
            if file_path.is_file():
                file_path.unlink()
        for directory in sorted(cleaned_root.rglob("*"), reverse=True):
            if directory.is_dir():
                directory.rmdir()
    cleaned_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    skipped = 0
    reductions: list[float] = []
    samples: list[tuple[str, str]] = []

    for row in rows:
        processed += 1
        source_path = row["source_path"]
        absolute_path = corpus_root / source_path
        original_text = absolute_path.read_text(encoding="utf-8", errors="replace")
        cleaned_text = clean_document(text=original_text, config=config)

        if len(cleaned_text) < config.min_cleaned_chars:
            skipped += 1
            continue

        _write_text(path=cleaned_root / source_path, content=cleaned_text)
        written += 1
        reductions.append(
            max(0.0, 1 - (len(cleaned_text) / max(1, len(original_text))))
        )
        samples.append((original_text, cleaned_text))

    random.seed(42)
    random.shuffle(samples)
    validation = samples[: config.validation_sample_size]
    damage_scores = [
        semantic_damage_ratio(
            original_text=original,
            cleaned_text=cleaned,
            min_paragraph_chars=config.min_semantic_paragraph_chars,
            overlap_threshold=config.semantic_overlap_threshold,
        )
        for original, cleaned in validation
    ]
    mean_damage = sum(damage_scores) / len(damage_scores) if damage_scores else 0.0
    max_damage = max(damage_scores) if damage_scores else 0.0
    payload = {
        "processed_files": processed,
        "written_files": written,
        "skipped_files": skipped,
        "mean_size_reduction_pct": (sum(reductions) / len(reductions) * 100)
        if reductions
        else 0.0,
        "validation_sample_size": len(validation),
        "mean_semantic_damage_ratio": mean_damage,
        "max_semantic_damage_ratio_observed": max_damage,
        "max_semantic_damage_ratio_target": config.max_semantic_damage_ratio,
        "threshold_passed": max_damage < config.max_semantic_damage_ratio,
    }

    _write_json(path=qa_json_path, payload=payload)
    _build_report_markdown(payload=payload, output_path=summary_md_path)
    print(f"Processed files: {processed}")
    print(f"Written files: {written}")
    print(f"Threshold passed: {payload['threshold_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

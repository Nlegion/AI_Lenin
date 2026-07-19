#!/usr/bin/env python3
"""Run rollback drill across retrieval migration modes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import json
from pathlib import Path
import tempfile
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.rag_system import get_rag_system  # noqa: E402
from src.core.retrieval.provider_factory import build_provider  # noqa: E402


def _queries() -> list[str]:
    return [
        "Рост инфляции и безработицы",
        "Международные санкции и торговые ограничения",
        "Классовые противоречия в экономическом кризисе",
        "Политика правительства по зарплатам и труду",
        "Империализм и глобальный рынок капитала",
    ]


def _provider_for_mode(config_path: Path, mode: str):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("retrieval_pipeline", payload)
    migration = section.get("migration", {})
    migration["mode"] = mode
    section["migration"] = migration
    payload["retrieval_pipeline"] = section

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml", encoding="utf-8") as file_handle:
        temp_path = Path(file_handle.name)
        file_handle.write(yaml.safe_dump(payload, allow_unicode=True))

    provider = build_provider(
        config_path=temp_path,
        base_dir=REPO_ROOT,
        rag_system=get_rag_system(),
    )
    temp_path.unlink(missing_ok=True)
    return provider


def _close_provider(provider) -> None:
    # Local Qdrant backend uses filesystem locks; release clients between modes.
    if provider is None:
        return
    for attr in ("client",):
        client = getattr(provider, attr, None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
    for attr in ("primary", "shadow"):
        nested = getattr(provider, attr, None)
        if nested is not None and nested is not provider:
            _close_provider(nested)
    gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval migration rollback drill.")
    parser.add_argument("--config", default="config/retrieval_pipeline.yaml")
    parser.add_argument("--out-json", default=".cursor/artifacts/retrieval/rollback_drill.json")
    parser.add_argument("--out-md", default=".cursor/artifacts/retrieval/rollback_drill.md")
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    modes = ["ab_shadow", "qdrant_only", "chroma_only"]
    queries = _queries()
    results: dict[str, dict[str, float | int]] = {}

    for mode in modes:
        provider = _provider_for_mode(config_path=config_path, mode=mode)
        non_empty = 0
        for query in queries:
            outcome = provider.retrieve_context(query_text=query, author_filter="Ленин")
            if outcome.context.strip():
                non_empty += 1
        results[mode] = {
            "queries_total": len(queries),
            "non_empty_context": non_empty,
            "non_empty_ratio": non_empty / len(queries),
        }
        _close_provider(provider=provider)

    payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }
    out_json = (REPO_ROOT / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = (REPO_ROOT / args.out_md).resolve()
    lines = [
        "# Rollback Drill Summary",
        "",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        "",
        "| Mode | Non-empty contexts | Total | Ratio |",
        "|---|---:|---:|---:|",
    ]
    for mode in modes:
        row = results[mode]
        lines.append(
            f"| `{mode}` | {row['non_empty_context']} | {row['queries_total']} | {row['non_empty_ratio']:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("rollback_drill_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

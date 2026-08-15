#!/usr/bin/env python3
"""Build legacy RAG inventory report for controlled archival."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.settings.legacy_registry import load_legacy_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build legacy RAG inventory report.")
    parser.add_argument("--config", default="config/legacy_rag_components.yaml")
    parser.add_argument(
        "--out-md", default=".cursor/artifacts/legacy/legacy_rag_inventory.md"
    )
    parser.add_argument(
        "--out-json", default=".cursor/artifacts/legacy/legacy_rag_inventory.json"
    )
    args = parser.parse_args()

    registry = load_legacy_registry(path=(REPO_ROOT / args.config).resolve())
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for component in registry.components:
        full_path = (REPO_ROOT / component.path).resolve()
        rows.append(
            {
                "path": component.path,
                "category": component.category,
                "status": component.status,
                "action": component.action,
                "exists": full_path.exists(),
                "note": component.note,
            }
        )

    out_json = (REPO_ROOT / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": generated_at,
        "policy_version": registry.policy_version,
        "components": rows,
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out_md = (REPO_ROOT / args.out_md).resolve()
    lines = [
        "# Legacy RAG Inventory",
        "",
        f"- Generated at (UTC): {generated_at}",
        f"- Policy version: `{registry.policy_version}`",
        "",
        "| Path | Category | Status | Action | Exists |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['path']}` | `{row['category']}` | `{row['status']}` | `{row['action']}` | `{row['exists']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Components tracked: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

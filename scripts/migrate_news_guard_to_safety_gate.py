"""Migrate policy keys from news_guard.yaml into safety_gate_config.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_POLICY_KEYS = (
    "allow_topics",
    "hard_deny_topics",
    "quarantine_topics",
    "hard_deny_keywords",
    "quarantine_keywords",
    "military_topics",
    "public_interest_topics",
    "economy_policy_markers",
    "yellow_block_patterns",
    "refusal_message",
    "skip_message",
    "classify_on_unknown_as",
    "block_private_pii",
)


def migrate(*, news_guard_path: Path, safety_gate_path: Path, dry_run: bool = False) -> dict:
    ng_payload = yaml.safe_load(news_guard_path.read_text(encoding="utf-8")) or {}
    input_gate = (ng_payload.get("news_guard") or ng_payload).get("input_gate") or {}
    sg_payload: dict = {}
    if safety_gate_path.is_file():
        sg_payload = yaml.safe_load(safety_gate_path.read_text(encoding="utf-8")) or {}
    section = dict(sg_payload.get("safety_gate") or sg_payload or {})
    flags = dict(section.get("flags") or {})
    policy = dict(section.get("policy") or {})
    copied: list[str] = []
    for key in _POLICY_KEYS:
        if key in input_gate:
            policy[key] = input_gate[key]
            copied.append(key)
    section["flags"] = flags
    section["policy"] = policy
    out = {"safety_gate": section}
    if not dry_run:
        safety_gate_path.parent.mkdir(parents=True, exist_ok=True)
        safety_gate_path.write_text(
            yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return {"copied_keys": copied, "dry_run": dry_run, "path": str(safety_gate_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate NewsGuard policy into SafetyGate config")
    parser.add_argument(
        "--news-guard",
        default=str(REPO_ROOT / "config" / "news_guard.yaml"),
    )
    parser.add_argument(
        "--safety-gate",
        default=str(REPO_ROOT / "config" / "safety_gate_config.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = migrate(
        news_guard_path=Path(args.news_guard),
        safety_gate_path=Path(args.safety_gate),
        dry_run=bool(args.dry_run),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

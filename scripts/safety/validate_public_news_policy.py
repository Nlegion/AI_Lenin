#!/usr/bin/env python3
"""Validate mandatory disclaimer policy for public mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate public NewsGuard policy.")
    parser.add_argument("--config", default="config/news_guard.yaml")
    parser.add_argument("--public-mode", action="store_true", default=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    config_path = (repo / args.config).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("news_guard", payload)
    disclaimer = section.get("disclaimer", {})

    if args.public_mode:
        if not bool(disclaimer.get("enabled", False)):
            print("policy_error: disclaimer must be enabled in public mode")
            return 1
        text = str(disclaimer.get("text", "")).strip()
        if not text:
            print("policy_error: disclaimer text must be non-empty")
            return 1

    print("public_policy_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

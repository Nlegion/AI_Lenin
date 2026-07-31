"""Restore last-stable NewsGuard-related configs from config/stable/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STABLE = REPO / "config" / "stable"
DEFAULT_FILES = ("news_guard.yaml",)


def snapshot(*, files: list[str]) -> None:
    STABLE.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = REPO / "config" / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, STABLE / name)
        print(f"snapshotted {src} -> {STABLE / name}")


def restore(*, files: list[str]) -> None:
    if not STABLE.exists():
        raise FileNotFoundError(f"Missing stable dir: {STABLE}")
    for name in files:
        src = STABLE / name
        if not src.exists():
            raise FileNotFoundError(src)
        dest = REPO / "config" / name
        shutil.copy2(src, dest)
        print(f"restored {src} -> {dest}")
    print(
        "Manual next step: restart the application "
        "(e.g. python src/main.py or your service unit). Auto-restart is disabled."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot/restore stable gate configs")
    parser.add_argument("action", choices=("snapshot", "restore"))
    parser.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    args = parser.parse_args()
    if args.action == "snapshot":
        snapshot(files=list(args.files))
    else:
        restore(files=list(args.files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

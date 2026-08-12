#!/usr/bin/env python3
"""Bootstrap dev environment with runtime + quality dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

def _run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)  # nosec B603
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap AI_Lenin dev environment.")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-dev", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    python = sys.executable

    if not args.skip_runtime:
        _run([python, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo)
    if not args.skip_dev:
        _run([python, "-m", "pip", "install", "-r", "requirements-dev.txt"], cwd=repo)

    print("bootstrap_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

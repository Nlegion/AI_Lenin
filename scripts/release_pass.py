#!/usr/bin/env python3
"""Release-pass gate runner for clean environment validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full release-pass gates.")
    parser.add_argument("--skip-security-m", action="store_true", help="Skip NewsGuard regression check.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    python = sys.executable

    _run([python, "scripts/run_subplan_gates.py", "--run-optional"], cwd=repo)
    _run([python, "-m", "pytest", "tests/test_local_rag_dryrun.py", "-q"], cwd=repo)
    if not args.skip_security_m:
        _run(
            [
                python,
                "scripts/evaluate_news_guard.py",
                "--config",
                "config/news_guard.yaml",
                "--out-json",
                ".cursor/artifacts/safety/news_guard_eval_release.json",
                "--out-md",
                ".cursor/artifacts/safety/news_guard_eval_release.md",
            ],
            cwd=repo,
        )
    print("release_pass_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

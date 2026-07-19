#!/usr/bin/env python3
"""Run mandatory quality gates for a refactor subplan."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    elapsed_seconds: float


def _run(command: list[str], cwd: Path) -> CommandResult:
    started_at = time()
    process = subprocess.run(command, cwd=cwd, check=False)
    elapsed = time() - started_at
    return CommandResult(command=command, exit_code=process.returncode, elapsed_seconds=elapsed)


def _format_command(command: list[str]) -> str:
    return " ".join(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Alembic + pytest gates for a subplan.")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help='Optional pytest -k pattern for targeted validation (runs in addition to full tests).',
    )
    parser.add_argument(
        "--run-optional",
        action="store_true",
        help="Also run optional checks when tools are installed (ruff, bandit).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    python_executable = sys.executable
    commands: list[list[str]] = [
        [python_executable, "-m", "alembic", "upgrade", "head"],
        [python_executable, "-m", "pytest", "tests", "-q"],
    ]
    if args.pattern:
        commands.append([python_executable, "-m", "pytest", "tests", "-k", args.pattern, "-q"])

    if args.run_optional:
        try:
            __import__("ruff")
            commands.extend(
                [
                    [python_executable, "-m", "ruff", "check", "src", "scripts", "tests"],
                    [python_executable, "-m", "ruff", "format", "--check", "src", "scripts", "tests"],
                ]
            )
        except ModuleNotFoundError:
            pass
        try:
            __import__("bandit")
            commands.append([python_executable, "-m", "bandit", "-r", "src", "scripts"])
        except ModuleNotFoundError:
            pass
        try:
            __import__("vulture")
            commands.append(
                [
                    python_executable,
                    "-m",
                    "vulture",
                    "src",
                    "scripts",
                    "tests",
                    "--min-confidence",
                    "100",
                ]
            )
        except ModuleNotFoundError:
            pass

    print("Running subplan gates from:", repo_root)
    overall_success = True
    results: list[CommandResult] = []
    for command in commands:
        print(f"\n>>> {_format_command(command)}")
        result = _run(command=command, cwd=repo_root)
        results.append(result)
        if result.exit_code != 0:
            overall_success = False
            print(
                f"FAILED ({result.exit_code}) in {result.elapsed_seconds:.2f}s: "
                f"{_format_command(command)}"
            )
            break
        print(f"PASSED in {result.elapsed_seconds:.2f}s")

    print("\nGate summary:")
    for result in results:
        status = "PASS" if result.exit_code == 0 else "FAIL"
        print(f"- [{status}] {_format_command(result.command)} ({result.elapsed_seconds:.2f}s)")

    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())

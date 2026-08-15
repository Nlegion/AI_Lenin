#!/usr/bin/env python3
"""Release-pass gate runner. CLI flags override/supplement config/release_gates.yaml."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.settings.release_gates import load_release_gates  # noqa: E402


def _run(command: list[str], *, cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)  # nosec B603
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def _write_override(reason: str, *, repo: Path) -> None:
    path = repo / ".cursor/artifacts/evaluation/rag_quality_override.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reason": reason,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run release-pass gates. CLI flags override/supplement config/release_gates.yaml."
        )
    )
    parser.add_argument(
        "--gates-config",
        default="config/release_gates.yaml",
        help="Unified release gates config.",
    )
    parser.add_argument(
        "--skip-rag-quality",
        action="store_true",
        help="Skip RAG quality gate (overrides YAML enabled).",
    )
    parser.add_argument(
        "--skip-security-m",
        action="store_true",
        help="Skip NewsGuard regression check.",
    )
    parser.add_argument(
        "--skip-anti-cliche",
        action="store_true",
        help="Skip anti-cliche eval even if YAML enables it.",
    )
    parser.add_argument(
        "--override-rag-quality",
        metavar="REASON",
        default=None,
        help="Log reason and treat RAG quality gate as passed.",
    )
    parser.add_argument(
        "--check-news-guard-delta",
        action="store_true",
        help="Compare NewsGuard eval to baseline (bootstraps baseline if missing).",
    )
    parser.add_argument(
        "--skip-subplan",
        action="store_true",
        help="Skip run_subplan_gates + dry-run pytest (for unit tests).",
    )
    args = parser.parse_args(argv)

    repo = REPO_ROOT
    python = sys.executable
    gates = load_release_gates(path=str((repo / args.gates_config).resolve()))

    try:
        if not args.skip_subplan:
            _run(
                [python, "scripts/ops/run_subplan_gates.py", "--run-optional"], cwd=repo
            )
            _run(
                [python, "-m", "pytest", "tests/test_local_rag_dryrun.py", "-q"],
                cwd=repo,
            )

        run_rag = gates.rag_quality.enabled and not args.skip_rag_quality
        if run_rag:
            if args.override_rag_quality:
                _write_override(args.override_rag_quality, repo=repo)
                print(f"rag_quality_overridden: {args.override_rag_quality}")
            else:
                _run(
                    [
                        python,
                        "scripts/retrieval/evaluate_rag_quality.py",
                        "--thresholds-config",
                        args.gates_config,
                        "--output-json",
                        ".cursor/artifacts/evaluation/rag_quality_metrics.json",
                        "--output-md",
                        ".cursor/artifacts/evaluation/rag_quality_summary.md",
                    ],
                    cwd=repo,
                )

        run_news = gates.news_guard_enabled and not args.skip_security_m
        news_out = ".cursor/artifacts/evaluation/news_guard_eval_release.json"
        news_md = ".cursor/artifacts/evaluation/news_guard_eval_release.md"
        if run_news:
            _run(
                [
                    python,
                    "scripts/safety/evaluate_news_guard.py",
                    "--config",
                    "config/news_guard.yaml",
                    "--out-json",
                    news_out,
                    "--out-md",
                    news_md,
                ],
                cwd=repo,
            )

        run_anti = gates.anti_cliche_enabled and not args.skip_anti_cliche
        if run_anti:
            _run([python, "scripts/quality/evaluate_anti_cliche.py"], cwd=repo)

        check_delta = args.check_news_guard_delta or gates.news_guard_delta_enabled
        if check_delta:
            baseline = (repo / gates.news_guard_baseline_json).resolve()
            current = (repo / news_out).resolve()
            if not run_news:
                _run(
                    [
                        python,
                        "scripts/safety/evaluate_news_guard.py",
                        "--config",
                        "config/news_guard.yaml",
                        "--out-json",
                        news_out,
                        "--out-md",
                        news_md,
                    ],
                    cwd=repo,
                )
            if not baseline.is_file():
                baseline.parent.mkdir(parents=True, exist_ok=True)
                baseline.write_text(
                    current.read_text(encoding="utf-8"), encoding="utf-8"
                )
                print(f"news_guard_baseline_bootstrapped: {baseline}")
            else:
                base_payload = json.loads(baseline.read_text(encoding="utf-8"))
                cur_payload = json.loads(current.read_text(encoding="utf-8"))
                # Soft compare: fail if top-level summary keys regress when present
                base_fail = int(
                    base_payload.get("failed", base_payload.get("failures", 0)) or 0
                )
                cur_fail = int(
                    cur_payload.get("failed", cur_payload.get("failures", 0)) or 0
                )
                if cur_fail > base_fail:
                    raise RuntimeError(
                        f"news_guard_delta_regression: failures {cur_fail} > baseline {base_fail}"
                    )

    except Exception as exc:
        print(f"release_pass_error: {exc}", file=sys.stderr)
        return 1

    print("release_pass_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

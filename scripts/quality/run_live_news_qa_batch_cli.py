"""CLI for live-news QA batch (real RSS, no Telegram)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch live news (TASS RSS), generate Lenin answers without Telegram. "
            "Safety rejects are logged but do not count toward --target-done."
        )
    )
    parser.add_argument(
        "--target-done", type=int, default=50, help="Successful LLM answers to collect."
    )
    parser.add_argument(
        "--fetch-limit", type=int, default=0, help="Max RSS items to consider (0=all)."
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=300.0,
        help=(
            "If done < target after one RSS pass, sleep this many seconds and refetch "
            "(0 = fail when pool exhausted, legacy one-shot)."
        ),
    )
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=24.0,
        help="Stop polling after this many hours even if target-done not reached (0=unlimited).",
    )
    parser.add_argument(
        "--persona-model", choices=["base_strong"], default="base_strong"
    )
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--start-wait", type=float, default=300.0)
    parser.add_argument("--allow-legacy-fallback", action="store_true")
    parser.add_argument("--require-rag-nonempty", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=".cursor/artifacts/quality")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--txt-max-chars", type=int, default=0)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--save-full-prompts", action="store_true")
    parser.add_argument("--generation-config", default="config/generation.yaml")
    parser.add_argument("--news-guard-config", default="config/news_guard.yaml")
    parser.add_argument(
        "--stem",
        default="live_news_qa",
        help="Artifact filename stem (timestamp appended).",
    )
    parser.add_argument(
        "--unknown-as-allow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy NewsGuard soft-pass (default: false). Ignored when PreRagCensor is active; "
            "use --censor-strict-review to reject yellow/review items."
        ),
    )
    parser.add_argument(
        "--censor-strict-review",
        action="store_true",
        default=False,
        help="Reject PreRagCensor review/yellow decisions instead of yellow generation (parity off).",
    )
    return parser

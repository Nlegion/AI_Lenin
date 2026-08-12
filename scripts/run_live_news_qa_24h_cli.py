"""CLI for 24h continuous live-news QA (TASS poll loop, no Telegram)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Poll TASS RSS for --duration-hours, generate Lenin answers without Telegram. "
            "Successful answers append to *.txt; safety rejects to *.rejected.txt."
        )
    )
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=900.0,
        help="Seconds between RSS fetch cycles (default 15 min).",
    )
    parser.add_argument(
        "--max-per-cycle",
        type=int,
        default=20,
        help="Max new items to process per fetch cycle (0=unlimited).",
    )
    parser.add_argument("--fetch-limit", type=int, default=0, help="Max RSS items per fetch (0=all).")
    parser.add_argument("--persona-model", choices=["base_strong", "fine_tuned"], default="base_strong")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--start-wait", type=float, default=300.0)
    parser.add_argument("--allow-legacy-fallback", action="store_true")
    parser.add_argument("--require-rag-nonempty", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Resume from existing checkpoint JSONL.")
    parser.add_argument("--output-dir", default=".cursor/artifacts/quality")
    parser.add_argument("--force", action="store_true", help="Wipe artifacts and start fresh.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--txt-max-chars", type=int, default=0)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--save-full-prompts", action="store_true")
    parser.add_argument("--generation-config", default="config/generation.yaml")
    parser.add_argument("--news-guard-config", default="config/news_guard.yaml")
    parser.add_argument("--stem", default="live_news_qa_24h")
    parser.add_argument(
        "--unknown-as-allow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy NewsGuard soft-pass (default false; ignored when PreRagCensor is active).",
    )
    parser.add_argument(
        "--censor-strict-review",
        action="store_true",
        default=False,
        help="Reject PreRagCensor review/yellow decisions instead of yellow generation.",
    )
    return parser

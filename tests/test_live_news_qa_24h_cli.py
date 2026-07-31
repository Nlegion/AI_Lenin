"""Smoke tests for 24h live QA CLI."""

from __future__ import annotations

from scripts.run_live_news_qa_24h_cli import build_parser


def test_24h_cli_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.duration_hours == 24.0
    assert args.poll_seconds == 900.0
    assert args.stem == "live_news_qa_24h"
    assert args.unknown_as_allow is True

"""CLI for R1/R2/R3 example trace (news + slots + LLM answer)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump news + R1/R2/R3 retrieval + LLM answer for N items."
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Successful LLM examples."
    )
    parser.add_argument("--fetch-limit", type=int, default=0, help="RSS cap (0=all).")
    parser.add_argument("--from-jsonl", default=None, help="Replay QaItem JSONL.")
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--start-wait", type=float, default=300.0)
    parser.add_argument("--allow-legacy-fallback", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument(
        "--persona-model", choices=["base_strong"], default="base_strong"
    )
    parser.add_argument("--generation-config", default="config/generation.yaml")
    parser.add_argument("--output-dir", default=".cursor/artifacts/quality")
    parser.add_argument("--stem", default="r13_example_trace")
    return parser

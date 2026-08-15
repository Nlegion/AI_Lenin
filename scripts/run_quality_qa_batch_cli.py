"""CLI argument parser for quality QA batch."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quality QA batch without Telegram.")
    parser.add_argument("--input", default="data/eval/quality_qa_batch.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--persona-model", choices=["base_strong"], default="base_strong")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--start-wait", type=float, default=120.0)
    parser.add_argument("--allow-legacy-fallback", action="store_true")
    parser.add_argument("--guard-check-only", action="store_true")
    parser.add_argument("--max-blocked-ratio", type=float, default=0.0)
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint JSONL file.")
    parser.add_argument("--output-dir", default=".cursor/artifacts/quality")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--txt-max-chars", type=int, default=0)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--save-full-prompts", action="store_true")
    parser.add_argument("--require-rag-nonempty", action="store_true")
    parser.add_argument("--pre-gate-only", action="store_true", help="Apply NewsGuard pre-LLM only; no llama-server.")
    parser.add_argument("--generation-config", default="config/generation.yaml")
    parser.add_argument("--news-guard-config", default="config/news_guard.yaml")
    return parser

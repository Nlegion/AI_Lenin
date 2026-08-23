#!/usr/bin/env python3
"""Collect N news examples: incoming item, R1/R2/R3 chunks, LLM answer."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib._live_news_qa_censor import (  # noqa: E402
    apply_live_pre_rag_gate,
    build_live_pre_rag_censor,
)
from scripts.lib._live_news_qa_fetch import fetch_live_qa_items  # noqa: E402
from scripts.lib._quality_qa_io import QaItem  # noqa: E402
from scripts.lib._quality_qa_runtime import base_row, generate_one  # noqa: E402
from scripts.lib._r13_example_report import (  # noqa: E402
    load_fixture_qa_items,
    load_jsonl_qa_items,
    write_report_files,
)
from scripts.quality.run_r13_example_trace_cli import build_parser  # noqa: E402
from src.core.lenin_analyzer import LeninAnalyzer  # noqa: E402
from src.core.llama_server import LeninServer  # noqa: E402
from src.core.safety.news_guard import NewsGuard, load_news_guard_config  # noqa: E402
from src.core.settings.device import is_llama_server_active  # noqa: E402
from src.core.settings.generation_config import load_generation_config  # noqa: E402

logger = logging.getLogger("r13_example_trace")


def _load_items(args: argparse.Namespace) -> list[QaItem]:
    if args.from_jsonl:
        jsonl = Path(args.from_jsonl)
        if not jsonl.is_absolute():
            jsonl = ROOT / jsonl
        return load_jsonl_qa_items(path=jsonl, limit=0)
    if args.fixtures:
        return load_fixture_qa_items(
            path=ROOT / "config" / "dryrun_fixtures.yaml", limit=0
        )
    return fetch_live_qa_items(fetch_limit=int(args.fetch_limit))


async def _ensure_server(
    args: argparse.Namespace, generation_config: object
) -> LeninServer | None:
    spawn_local = bool(generation_config.spawn_local)
    server_url = generation_config.server_url
    if not spawn_local:
        logger.info("Remote LLM mode url=%s", server_url)
        return None
    if is_llama_server_active(server_url=server_url):
        logger.warning("Server already running on %s", server_url)
        return None
    if not args.start_server:
        raise RuntimeError("LLM not reachable; pass --start-server")
    owned = LeninServer(
        persona_model=generation_config.persona_model,
        generation_config=generation_config,
    )
    if not await owned.start_server():
        raise RuntimeError("Failed to start llama-server")
    deadline = time.monotonic() + float(args.start_wait)
    while time.monotonic() < deadline:
        if is_llama_server_active(server_url=server_url):
            return owned
        await asyncio.sleep(2.0)
    await owned.stop_server()
    raise RuntimeError("llama-server not healthy within --start-wait")


async def async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = max(1, int(args.limit))
    items = _load_items(args)
    if not items:
        logger.error("No news items loaded")
        return 2
    generation_config = load_generation_config(path=ROOT / args.generation_config)
    generation_config = generation_config.with_persona_model(args.persona_model)
    owned: LeninServer | None = None
    try:
        owned = await _ensure_server(args, generation_config)
    except RuntimeError as error:
        logger.error("%s", error)
        return 3
    analyzer = LeninAnalyzer(persona_model=generation_config.persona_model)
    analyzer.generation_config = generation_config
    analyzer.session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=float(args.llm_timeout),
            sock_connect=min(30.0, float(args.llm_timeout)),
        )
    )
    if analyzer.retrieval_provider is None and not args.allow_legacy_fallback:
        logger.error("Retrieval unavailable; use --allow-legacy-fallback")
        await analyzer.close_session()
        if owned is not None:
            await owned.stop_server()
        return 4
    guard = NewsGuard(
        config=load_news_guard_config(path=ROOT / "config" / "news_guard.yaml")
    )
    censor = build_live_pre_rag_censor(base_dir=ROOT, news_guard=guard)
    pipeline = analyzer._get_pipeline()  # noqa: SLF001
    rows: list[dict] = []
    done = 0
    try:
        for item in items:
            if done >= target:
                break
            row = base_row(
                item,
                persona_model=generation_config.persona_model,
                input_hash=item.input_hash(),
            )
            outcome = await apply_live_pre_rag_gate(censor=censor, item=item, row=row)
            if outcome.blocked_row is not None:
                if args.include_blocked:
                    rows.append(outcome.blocked_row)
                logger.info("blocked id=%s", item.id)
                continue
            gen_ctx = outcome.generation
            row = await generate_one(
                analyzer=analyzer,
                pipeline=pipeline,
                item=item,
                retries=int(args.retries),
                save_full_prompts=False,
                skip_input_gate=True,
                risk_tier=gen_ctx.risk_tier if gen_ctx else "green",
                context_hints=gen_ctx.context_hints if gen_ctx else None,
                needs_yellow_warning=bool(gen_ctx and gen_ctx.needs_yellow_warning),
            )
            if row.get("status") == "done" and not row.get("blocked"):
                rows.append(row)
                done += 1
                logger.info(
                    "done id=%s r1=%s r2=%s r3=%s progress=%s/%s",
                    item.id,
                    row.get("r1_count"),
                    row.get("r2_count"),
                    row.get("r3_count"),
                    done,
                    target,
                )
            elif args.include_blocked:
                rows.append(row)
    finally:
        await pipeline.close()
        await analyzer.close_session()
        if owned is not None:
            await owned.stop_server()
    md_path, jsonl_path = write_report_files(
        output_dir=(ROOT / args.output_dir).resolve(),
        stem=str(args.stem),
        rows=rows,
    )
    logger.info("wrote md=%s jsonl=%s examples=%s", md_path, jsonl_path, len(rows))
    if done < target:
        logger.error("Incomplete: done=%s < target=%s", done, target)
        return 6
    return 0


def main() -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(async_main(args=build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

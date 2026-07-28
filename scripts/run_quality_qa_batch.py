#!/usr/bin/env python3
"""Batch quality QA generation without Telegram publish."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import aiohttp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._quality_qa_io import (  # noqa: E402
    append_jsonl,
    format_txt_block,
    format_txt_header,
    load_checkpoint_last_wins,
    load_qa_items,
    resolve_artifact_paths,
    should_skip_checkpoint_row,
)
from scripts._quality_qa_runtime import (  # noqa: E402
    base_row,
    generate_one,
    rag_probe,
    run_guard_check,
)
from scripts.run_quality_qa_batch_cli import build_parser  # noqa: E402
from src.core.lenin_analyzer import LeninAnalyzer  # noqa: E402
from src.core.llama_server import LeninServer  # noqa: E402
from src.core.safety.news_guard import NewsGuard, load_news_guard_config  # noqa: E402
from src.core.settings.device import is_llama_server_active  # noqa: E402
from src.core.settings.generation_config import load_generation_config  # noqa: E402

logger = logging.getLogger("quality_qa_batch")


def _configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)


async def _wait_for_health(*, server_url: str, start_wait: float) -> bool:
    deadline = time.monotonic() + start_wait
    while time.monotonic() < deadline:
        if is_llama_server_active(server_url=server_url, timeout_sec=1.0):
            return True
        await asyncio.sleep(2.0)
    return False


async def async_main(args: argparse.Namespace) -> int:
    _configure_logging(Path(args.log_file) if args.log_file else None)
    if args.require_rag_nonempty and args.allow_legacy_fallback:
        logger.error("Incompatible flags: --require-rag-nonempty with --allow-legacy-fallback")
        return 2

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()
    items = load_qa_items(path=input_path)
    if args.limit is not None:
        items = items[: max(0, int(args.limit))]
    if not items:
        logger.error("No items after --limit")
        return 2

    guard = NewsGuard(config=load_news_guard_config(path=REPO_ROOT / args.news_guard_config))
    if args.guard_check_only:
        return run_guard_check(items=items, guard=guard, max_blocked_ratio=args.max_blocked_ratio)

    generation_config = load_generation_config(path=REPO_ROOT / args.generation_config)
    try:
        if args.persona_model:
            generation_config = generation_config.with_persona_model(args.persona_model)
        backend = generation_config.active_backend()
    except Exception as error:  # noqa: BLE001
        logger.error("Invalid persona_model/config: %s", error)
        return 2

    model_path = (REPO_ROOT / backend.model_path).resolve()
    if not model_path.exists():
        logger.error("Model file missing: %s", model_path)
        return 2

    server_url = generation_config.server_url
    owned_server: LeninServer | None = None
    if is_llama_server_active(server_url=server_url):
        logger.warning(
            "Server already running on %s. Ensure it serves persona_model=%s GGUF; "
            "otherwise stop it and re-run.",
            server_url,
            generation_config.persona_model,
        )
    elif args.start_server:
        owned_server = LeninServer(
            persona_model=generation_config.persona_model,
            generation_config=generation_config,
        )
        if not await owned_server.start_server():
            logger.error("Failed to start llama-server for persona_model=%s", generation_config.persona_model)
            return 3
        if not await _wait_for_health(server_url=server_url, start_wait=float(args.start_wait)):
            logger.error("llama-server did not become healthy within --start-wait=%s", args.start_wait)
            return 3
    else:
        logger.error("LLM not reachable at %s; pass --start-server or start llama-server", server_url)
        return 3

    checkpoint_arg = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_arg is not None and not checkpoint_arg.is_absolute():
        checkpoint_arg = (REPO_ROOT / checkpoint_arg).resolve()
    artifacts = resolve_artifact_paths(
        input_path=input_path,
        output_dir=(REPO_ROOT / args.output_dir).resolve(),
        checkpoint=checkpoint_arg,
    )
    if args.force:
        for path in (artifacts.checkpoint, artifacts.results, artifacts.txt):
            if path.exists():
                path.unlink()

    prior = {} if args.force else load_checkpoint_last_wins(path=artifacts.checkpoint)
    analyzer = LeninAnalyzer(persona_model=generation_config.persona_model)
    analyzer.generation_config = generation_config
    analyzer.session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=float(args.llm_timeout),
            sock_connect=min(30.0, float(args.llm_timeout)),
        )
    )

    if analyzer.retrieval_provider is None and not args.allow_legacy_fallback:
        logger.error("Qdrant/retrieval unavailable; re-run with --allow-legacy-fallback")
        await analyzer.close_session()
        return 4

    if args.allow_legacy_fallback:
        logger.info("RAG probe skipped due to legacy fallback")
    else:
        if rag_probe(analyzer=analyzer, item=items[0], require_nonempty=args.require_rag_nonempty) != 0:
            await analyzer.close_session()
            return 5

    pipeline = analyzer._get_pipeline()  # noqa: SLF001 - hot-path wiring
    artifacts.txt.parent.mkdir(parents=True, exist_ok=True)
    if not artifacts.txt.exists():
        artifacts.txt.write_text(format_txt_header(), encoding="utf-8")

    try:
        total = len(items)
        for index, item in enumerate(items, start=1):
            input_hash = item.input_hash()
            previous = prior.get(item.id)
            if should_skip_checkpoint_row(row=previous, input_hash=input_hash, force=args.force):
                logger.info("[%s/%s] %s status=skip", index, total, item.id)
                continue
            if previous is not None and str(previous.get("input_hash", "")) != input_hash:
                logger.warning("input_hash mismatch for %s — regenerating", item.id)

            gate = guard.evaluate_input(title=item.title, content=item.content, source=item.source or "unknown")
            if gate.decision in {"deny", "quarantine"}:
                row = base_row(item, persona_model=generation_config.persona_model, input_hash=input_hash)
                row["status"] = "blocked"
                row["blocked"] = True
                row["reason_codes"] = list(gate.reason_codes)
                row["answer"] = f"[BLOCKED: {gate.decision}] {gate.message}"
            else:
                row = await generate_one(
                    analyzer=analyzer,
                    pipeline=pipeline,
                    item=item,
                    retries=int(args.retries),
                    save_full_prompts=bool(args.save_full_prompts),
                )

            append_jsonl(path=artifacts.checkpoint, row=row)
            append_jsonl(path=artifacts.results, row=row)
            answer_txt = row.get("answer") or f"[ERROR: {row.get('error_type')}] {row.get('error')}"
            with artifacts.txt.open("a", encoding="utf-8") as handle:
                handle.write(
                    format_txt_block(
                        index=index,
                        item=item,
                        answer=str(answer_txt),
                        txt_max_chars=int(args.txt_max_chars),
                    )
                )
            logger.info("[%s/%s] %s status=%s", index, total, item.id, row["status"])
            prior[item.id] = row
    finally:
        await pipeline.close()
        await analyzer.close_session()
        if owned_server is not None:
            await owned_server.stop_server()

    logger.info("wrote checkpoint=%s results=%s txt=%s", artifacts.checkpoint, artifacts.results, artifacts.txt)
    return 0


def main() -> int:
    return asyncio.run(async_main(args=build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

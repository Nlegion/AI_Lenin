#!/usr/bin/env python3
"""Live-news QA batch: fetch TASS RSS, generate answers, no Telegram.

Safety deny/quarantine → *.rejected.jsonl / *.rejected.txt (not counted).
Successful LLM answers are numbered in *.txt toward --target-done (default 50).
Does not modify scripts/run_quality_qa_batch.py.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

import aiohttp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._live_news_qa_artifacts import (  # noqa: E402
    count_done_rows,
    rejected_paths,
    resolve_live_artifacts,
)
from scripts._live_news_qa_fetch import fetch_live_qa_items  # noqa: E402
from scripts._live_news_qa_gate import apply_live_pre_llm_gate  # noqa: E402
from scripts._quality_qa_io import (  # noqa: E402
    append_jsonl,
    format_txt_block,
    format_txt_header,
    load_checkpoint_last_wins,
    should_skip_checkpoint_row,
)
from scripts._quality_qa_runtime import (  # noqa: E402
    base_row,
    generate_one,
    rag_probe,
)
from scripts.run_live_news_qa_batch_cli import build_parser  # noqa: E402
from src.core.lenin_analyzer import LeninAnalyzer  # noqa: E402
from src.core.llama_server import LeninServer  # noqa: E402
from src.core.safety.news_guard import NewsGuard, load_news_guard_config  # noqa: E402
from src.core.settings.device import is_llama_server_active  # noqa: E402
from src.core.settings.generation_config import load_generation_config  # noqa: E402

logger = logging.getLogger("live_news_qa_batch")


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


def _ensure_headers(*, txt: Path, rejected_txt: Path) -> None:
    txt.parent.mkdir(parents=True, exist_ok=True)
    if not txt.exists():
        txt.write_text(
            format_txt_header()
            + "# Live news: numbered answers only (successful).\n"
            + "# Safety rejects → *.rejected.txt / *.rejected.jsonl (not counted).\n\n",
            encoding="utf-8",
        )
    if not rejected_txt.exists():
        rejected_txt.write_text(
            "# Safety rejects (deny/quarantine). Not counted toward target-done.\n\n",
            encoding="utf-8",
        )


async def async_main(args) -> int:
    _configure_logging(Path(args.log_file) if args.log_file else None)
    if args.require_rag_nonempty and args.allow_legacy_fallback:
        logger.error("Incompatible flags: --require-rag-nonempty with --allow-legacy-fallback")
        return 2
    target = max(1, int(args.target_done))
    items = fetch_live_qa_items(fetch_limit=int(args.fetch_limit))
    if not items:
        logger.error("No live news fetched from TASS RSS")
        return 2
    logger.info("fetched_live_news n=%s target_done=%s", len(items), target)

    guard = NewsGuard(config=load_news_guard_config(path=REPO_ROOT / args.news_guard_config))
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
        return 3

    checkpoint_arg = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_arg is not None and not checkpoint_arg.is_absolute():
        checkpoint_arg = (REPO_ROOT / checkpoint_arg).resolve()
    artifacts = resolve_live_artifacts(
        output_dir=(REPO_ROOT / args.output_dir).resolve(),
        stem=str(args.stem),
        checkpoint=checkpoint_arg,
    )
    rejected_jsonl, rejected_txt = rejected_paths(artifacts)
    if args.force:
        for path in (artifacts.checkpoint, artifacts.results, artifacts.txt, rejected_jsonl, rejected_txt):
            if path.exists():
                path.unlink()

    prior = {} if args.force else load_checkpoint_last_wins(path=artifacts.checkpoint)
    done_count = count_done_rows(prior)
    if done_count >= target:
        logger.info("Already have done=%s >= target=%s", done_count, target)
        return 0

    server_url = generation_config.server_url
    owned_server: LeninServer | None = None
    if is_llama_server_active(server_url=server_url):
        logger.warning("Server already running on %s", server_url)
    elif args.start_server:
        owned_server = LeninServer(
            persona_model=generation_config.persona_model,
            generation_config=generation_config,
        )
        if not await owned_server.start_server():
            logger.error("Failed to start llama-server")
            return 3
        if not await _wait_for_health(server_url=server_url, start_wait=float(args.start_wait)):
            logger.error("llama-server not healthy within --start-wait")
            return 3
    else:
        logger.error("LLM not reachable; pass --start-server")
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
        return 4
    if args.allow_legacy_fallback:
        logger.info("RAG probe skipped due to legacy fallback")
    elif rag_probe(analyzer=analyzer, item=items[0], require_nonempty=args.require_rag_nonempty) != 0:
        await analyzer.close_session()
        return 5

    pipeline = analyzer._get_pipeline()  # noqa: SLF001
    _ensure_headers(txt=artifacts.txt, rejected_txt=rejected_txt)
    rejected_count = 0
    error_count = 0
    try:
        for item in items:
            if done_count >= target:
                break
            input_hash = item.input_hash()
            previous = prior.get(item.id)
            if should_skip_checkpoint_row(row=previous, input_hash=input_hash, force=args.force):
                logger.info("skip id=%s", item.id)
                continue

            row = apply_live_pre_llm_gate(
                guard=guard,
                item=item,
                row=base_row(item, persona_model=generation_config.persona_model, input_hash=input_hash),
                unknown_as_allow=bool(args.unknown_as_allow),
            )
            if row is not None:
                rejected_count += 1
                append_jsonl(path=artifacts.checkpoint, row=row)
                append_jsonl(path=artifacts.results, row=row)
                append_jsonl(path=rejected_jsonl, row=row)
                with rejected_txt.open("a", encoding="utf-8") as handle:
                    handle.write(
                        format_txt_block(
                            index=rejected_count,
                            item=item,
                            answer=str(row.get("answer") or ""),
                            txt_max_chars=int(args.txt_max_chars),
                        )
                    )
                logger.info(
                    "rejected id=%s reason=%s done=%s/%s",
                    item.id,
                    row.get("skipped_llm_reason"),
                    done_count,
                    target,
                )
                prior[item.id] = row
                continue

            row = await generate_one(
                analyzer=analyzer,
                pipeline=pipeline,
                item=item,
                retries=int(args.retries),
                save_full_prompts=bool(args.save_full_prompts),
                skip_input_gate=True,
            )
            append_jsonl(path=artifacts.checkpoint, row=row)
            append_jsonl(path=artifacts.results, row=row)
            if row.get("status") == "done" and not row.get("blocked"):
                done_count += 1
                with artifacts.txt.open("a", encoding="utf-8") as handle:
                    handle.write(
                        format_txt_block(
                            index=done_count,
                            item=item,
                            answer=str(row.get("answer") or ""),
                            txt_max_chars=int(args.txt_max_chars),
                        )
                    )
                logger.info("done id=%s progress=%s/%s", item.id, done_count, target)
            else:
                error_count += 1
                logger.info("not_counted id=%s status=%s", item.id, row.get("status"))
            prior[item.id] = row
    finally:
        await pipeline.close()
        await analyzer.close_session()
        if owned_server is not None:
            await owned_server.stop_server()

    logger.info(
        "summary done=%s target=%s rejected=%s errors=%s txt=%s rejected_txt=%s",
        done_count,
        target,
        rejected_count,
        error_count,
        artifacts.txt,
        rejected_txt,
    )
    if done_count < target:
        logger.error("Pool exhausted: done=%s < target=%s (fetched=%s)", done_count, target, len(items))
        return 6
    return 0


def main() -> int:
    return asyncio.run(async_main(args=build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

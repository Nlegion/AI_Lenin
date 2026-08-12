#!/usr/bin/env python3
"""24h continuous live-news QA: poll TASS, generate answers, write txt for quality review.

Does not modify scripts/run_live_news_qa_batch.py. No Telegram.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._live_news_qa_24h_worker import process_live_item  # noqa: E402
from scripts._live_news_qa_artifacts import (  # noqa: E402
    count_done_rows,
    rejected_paths,
    resolve_live_artifacts,
)
from scripts._live_news_qa_censor import build_live_pre_rag_censor  # noqa: E402
from scripts._live_news_qa_fetch import fetch_live_qa_items  # noqa: E402
from scripts._quality_qa_io import (  # noqa: E402
    format_txt_header,
    load_checkpoint_last_wins,
    should_skip_checkpoint_row,
)
from scripts._quality_qa_runtime import rag_probe  # noqa: E402
from scripts.run_live_news_qa_24h_cli import build_parser  # noqa: E402
from src.core.lenin_analyzer import LeninAnalyzer  # noqa: E402
from src.core.llama_server import LeninServer  # noqa: E402
from src.core.safety.news_guard import NewsGuard, load_news_guard_config  # noqa: E402
from src.core.settings.device import is_llama_server_active  # noqa: E402
from src.core.settings.generation_config import load_generation_config  # noqa: E402

logger = logging.getLogger("live_news_qa_24h")


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


def _ensure_headers(*, txt: Path, rejected_txt: Path, duration_hours: float) -> None:
    txt.parent.mkdir(parents=True, exist_ok=True)
    if not txt.exists():
        txt.write_text(
            format_txt_header()
            + f"# Live 24h QA (duration_hours={duration_hours}). Numbered successful answers.\n"
            + "# Safety rejects → *.rejected.txt / *.rejected.jsonl.\n\n",
            encoding="utf-8",
        )
    if not rejected_txt.exists():
        rejected_txt.write_text(
            "# Safety rejects (deny/quarantine/skip). Not counted as successful answers.\n\n",
            encoding="utf-8",
        )


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def async_main(args) -> int:
    _configure_logging(Path(args.log_file) if args.log_file else None)
    if args.require_rag_nonempty and args.allow_legacy_fallback:
        logger.error("Incompatible flags: --require-rag-nonempty with --allow-legacy-fallback")
        return 2
    duration_s = max(60.0, float(args.duration_hours) * 3600.0)
    poll_s = max(30.0, float(args.poll_seconds))
    deadline = time.monotonic() + duration_s

    guard = NewsGuard(config=load_news_guard_config(path=REPO_ROOT / args.news_guard_config))
    try:
        censor = build_live_pre_rag_censor(
            base_dir=REPO_ROOT,
            news_guard=guard,
            disable_unknown_forward=True,
            enable_memory_cache=True,
        )
    except Exception as error:  # noqa: BLE001
        logger.error("Failed to initialize PreRagCensor: %s", error)
        return 2
    generation_config = load_generation_config(path=REPO_ROOT / args.generation_config)
    try:
        if args.persona_model:
            generation_config = generation_config.with_persona_model(args.persona_model)
        backend = generation_config.active_backend()
    except Exception as error:  # noqa: BLE001
        logger.error("Invalid persona_model/config: %s", error)
        return 2
    if not (REPO_ROOT / backend.model_path).resolve().exists():
        logger.error("Model file missing: %s", REPO_ROOT / backend.model_path)
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
    status_path = artifacts.results.with_suffix(".status.json")
    if args.force:
        for path in (artifacts.checkpoint, artifacts.results, artifacts.txt, rejected_jsonl, rejected_txt, status_path):
            if path.exists():
                path.unlink()

    prior = {} if args.force else load_checkpoint_last_wins(path=artifacts.checkpoint)
    counters = {"done": count_done_rows(prior), "rejected": 0, "errors": 0, "cycles": 0}

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

    pipeline = analyzer._get_pipeline()  # noqa: SLF001
    _ensure_headers(txt=artifacts.txt, rejected_txt=rejected_txt, duration_hours=float(args.duration_hours))
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("24h_qa_start duration_h=%s poll_s=%s txt=%s", args.duration_hours, poll_s, artifacts.txt)

    try:
        while time.monotonic() < deadline:
            counters["cycles"] += 1
            remaining_h = max(0.0, (deadline - time.monotonic()) / 3600.0)
            try:
                items = fetch_live_qa_items(fetch_limit=int(args.fetch_limit))
            except Exception as error:
                logger.exception("fetch_failed cycle=%s error=%s", counters["cycles"], error)
                items = []
            logger.info(
                "cycle=%s fetched=%s remaining_h=%.2f done=%s",
                counters["cycles"],
                len(items),
                remaining_h,
                counters["done"],
            )
            if items and counters["cycles"] == 1:
                if args.allow_legacy_fallback:
                    logger.info("RAG probe skipped due to legacy fallback")
                elif rag_probe(analyzer=analyzer, item=items[0], require_nonempty=args.require_rag_nonempty) != 0:
                    return 5

            processed_cycle = 0
            for item in items:
                if time.monotonic() >= deadline:
                    break
                if item.id in prior and should_skip_checkpoint_row(
                    row=prior.get(item.id),
                    input_hash=item.input_hash(),
                    force=False,
                ):
                    continue
                await process_live_item(
                    item=item,
                    censor=censor,
                    analyzer=analyzer,
                    pipeline=pipeline,
                    args=args,
                    prior=prior,
                    artifacts=artifacts,
                    rejected_jsonl=rejected_jsonl,
                    rejected_txt=rejected_txt,
                    counters=counters,
                )
                processed_cycle += 1
                if int(args.max_per_cycle) > 0 and processed_cycle >= int(args.max_per_cycle):
                    break

            _write_status(
                status_path,
                {
                    "started_at": started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "duration_hours": float(args.duration_hours),
                    "remaining_hours": round(remaining_h, 3),
                    "cycles": counters["cycles"],
                    "done": counters["done"],
                    "rejected": counters["rejected"],
                    "errors": counters["errors"],
                    "txt": str(artifacts.txt),
                    "rejected_txt": str(rejected_txt),
                    "running": True,
                },
            )
            if time.monotonic() >= deadline:
                break
            sleep_for = min(poll_s, max(1.0, deadline - time.monotonic()))
            logger.info("sleep_s=%.0f until next cycle", sleep_for)
            await asyncio.sleep(sleep_for)
    finally:
        await pipeline.close()
        await analyzer.close_session()
        if owned_server is not None:
            await owned_server.stop_server()
        _write_status(
            status_path,
            {
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "duration_hours": float(args.duration_hours),
                "remaining_hours": 0.0,
                "cycles": counters["cycles"],
                "done": counters["done"],
                "rejected": counters["rejected"],
                "errors": counters["errors"],
                "txt": str(artifacts.txt),
                "rejected_txt": str(rejected_txt),
                "running": False,
            },
        )

    logger.info(
        "summary_24h done=%s rejected=%s errors=%s cycles=%s txt=%s",
        counters["done"],
        counters["rejected"],
        counters["errors"],
        counters["cycles"],
        artifacts.txt,
    )
    return 0


def main() -> int:
    return asyncio.run(async_main(args=build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

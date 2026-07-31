"""Per-item processing for continuous live-news QA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from scripts._live_news_qa_gate import apply_live_pre_llm_gate
from scripts._quality_qa_io import append_jsonl, format_txt_block, should_skip_checkpoint_row
from scripts._quality_qa_runtime import base_row, generate_one
from src.core.lenin_analyzer import LeninAnalyzer
from src.core.safety.news_guard import NewsGuard

logger = logging.getLogger("live_news_qa_24h")


async def process_live_item(
    *,
    item,
    guard: NewsGuard,
    analyzer: LeninAnalyzer,
    pipeline,
    args,
    prior: dict[str, dict[str, Any]],
    artifacts,
    rejected_jsonl: Path,
    rejected_txt: Path,
    counters: dict[str, int],
) -> None:
    input_hash = item.input_hash()
    previous = prior.get(item.id)
    if should_skip_checkpoint_row(row=previous, input_hash=input_hash, force=False):
        logger.info("skip_seen id=%s", item.id)
        return

    row = apply_live_pre_llm_gate(
        guard=guard,
        item=item,
        row=base_row(item, persona_model=analyzer.generation_config.persona_model, input_hash=input_hash),
        unknown_as_allow=bool(args.unknown_as_allow),
    )
    if row is not None:
        counters["rejected"] += 1
        append_jsonl(path=artifacts.checkpoint, row=row)
        append_jsonl(path=artifacts.results, row=row)
        append_jsonl(path=rejected_jsonl, row=row)
        with rejected_txt.open("a", encoding="utf-8") as handle:
            handle.write(
                format_txt_block(
                    index=counters["rejected"],
                    item=item,
                    answer=str(row.get("answer") or ""),
                    txt_max_chars=int(args.txt_max_chars),
                )
            )
        logger.info(
            "rejected id=%s reason=%s done=%s",
            item.id,
            row.get("skipped_llm_reason"),
            counters["done"],
        )
        prior[item.id] = row
        return

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
        counters["done"] += 1
        with artifacts.txt.open("a", encoding="utf-8") as handle:
            handle.write(
                format_txt_block(
                    index=counters["done"],
                    item=item,
                    answer=str(row.get("answer") or ""),
                    txt_max_chars=int(args.txt_max_chars),
                )
            )
        logger.info("done id=%s progress=%s", item.id, counters["done"])
    else:
        counters["errors"] += 1
        logger.info("not_counted id=%s status=%s", item.id, row.get("status"))
    prior[item.id] = row

"""Per-item processing for continuous live-news QA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from scripts.lib._live_news_qa_censor import apply_live_pre_rag_gate
from scripts.lib._quality_qa_io import append_jsonl, format_txt_block, should_skip_checkpoint_row
from scripts.lib._quality_qa_runtime import base_row, generate_one
from src.core.lenin_analyzer import LeninAnalyzer
from src.core.safety.pre_rag_censor import PreRagCensor

logger = logging.getLogger("live_news_qa_24h")


async def process_live_item(
    *,
    item,
    censor: PreRagCensor,
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

    seed_row = base_row(
        item,
        persona_model=analyzer.generation_config.persona_model,
        input_hash=input_hash,
    )
    outcome = await apply_live_pre_rag_gate(
        censor=censor,
        item=item,
        row=seed_row,
        strict_review=bool(getattr(args, "censor_strict_review", False)),
    )
    if outcome.blocked_row is not None:
        row = outcome.blocked_row
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

    gen_ctx = outcome.generation
    row = await generate_one(
        analyzer=analyzer,
        pipeline=pipeline,
        item=item,
        retries=int(args.retries),
        save_full_prompts=bool(args.save_full_prompts),
        skip_input_gate=True,
        risk_tier=(gen_ctx.risk_tier if gen_ctx else "green"),
        context_hints=(gen_ctx.context_hints if gen_ctx else None),
        needs_yellow_warning=bool(gen_ctx.needs_yellow_warning) if gen_ctx else False,
    )
    if gen_ctx is not None:
        row["decision"] = gen_ctx.censor_decision
        row["censor_decision"] = gen_ctx.censor_decision
        row["censor_reason_codes"] = list(gen_ctx.censor_reason_codes)
        row["risk_tier"] = gen_ctx.risk_tier
        row["context_hints"] = list(gen_ctx.context_hints)
        row["needs_yellow_warning"] = bool(gen_ctx.needs_yellow_warning)
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

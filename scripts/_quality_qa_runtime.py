"""Runtime helpers for quality QA batch (guard, RAG probe, LLM generate, retries)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from scripts._quality_qa_io import QaItem, sha256_text
from src.core.generation.pipeline import AnalysisGenerationPipeline
from src.core.lenin_analyzer import LeninAnalyzer
from src.core.retrieval.migration_provider import MigrationRetrievalProvider
from src.core.safety.news_guard import NewsGuard

logger = logging.getLogger("quality_qa_batch")


def is_transient(error: Exception) -> bool:
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, aiohttp.ClientConnectorError)):
        return True
    if isinstance(error, aiohttp.ServerDisconnectedError):
        return True
    text = str(error).lower()
    if "429" in text or "http 5" in text:
        return True
    if "timeout" in text or "temporarily" in text:
        return True
    return False


def classify_error(error: Exception) -> str:
    text = str(error).lower()
    if "empty choices" in text or "empty_response" in text or "empty model content" in text:
        return "empty_response"
    if "invalid" in text and "json" in text:
        return "invalid_response"
    if is_transient(error):
        return "transient"
    return "runtime"


def probe_query(item: QaItem) -> str:
    lead = item.content.strip()
    if len(lead) >= 40:
        return lead[:500]
    return item.title.strip() or lead


def run_guard_check(*, items: list[QaItem], guard: NewsGuard, max_blocked_ratio: float) -> int:
    blocked_ids: list[str] = []
    counts = {"allow": 0, "warn": 0, "deny": 0, "quarantine": 0, "other": 0}
    for item in items:
        gate = guard.evaluate_input(title=item.title, content=item.content, source=item.source or "unknown")
        decision = str(gate.decision)
        counts[decision if decision in counts else "other"] += 1
        if decision in {"deny", "quarantine"}:
            blocked_ids.append(item.id)
            logger.info("guard id=%s decision=%s codes=%s", item.id, decision, ",".join(gate.reason_codes))
    total = max(len(items), 1)
    ratio = len(blocked_ids) / total
    logger.info(
        "guard_summary allow=%s warn=%s deny=%s quarantine=%s blocked_ratio=%.3f threshold=%.3f",
        counts["allow"],
        counts["warn"],
        counts["deny"],
        counts["quarantine"],
        ratio,
        max_blocked_ratio,
    )
    if blocked_ids:
        logger.info("blocked_ids=%s", ",".join(blocked_ids))
    return 1 if ratio > max_blocked_ratio else 0


def rag_probe(*, analyzer: LeninAnalyzer, item: QaItem, require_nonempty: bool) -> int:
    provider = analyzer.retrieval_provider
    if isinstance(provider, MigrationRetrievalProvider):
        provider = provider.primary
    query = probe_query(item=item)
    chunk_count = 0
    try:
        if provider is not None and hasattr(provider, "retrieve_with_trace"):
            candidates, _trace = provider.retrieve_with_trace(query_text=query, apply_judge=False)
            chunk_count = len(candidates or [])
        elif provider is not None and hasattr(provider, "retrieve_context"):
            result = provider.retrieve_context(query_text=query, author_filter="Ленин")
            context = str(getattr(result, "context", "") or "")
            chunk_count = 1 if context.strip() else 0
        else:
            logger.warning("RAG probe: retrieval provider unavailable")
    except Exception as error:  # noqa: BLE001
        logger.warning("RAG probe failed: %s", error)
        chunk_count = 0
    if chunk_count <= 0:
        logger.warning("RAG probe returned 0 chunks for query lead of %s", item.id)
        return 1 if require_nonempty else 0
    logger.info("RAG probe ok chunks=%s item=%s", chunk_count, item.id)
    return 0


def base_row(item: QaItem, *, persona_model: str, input_hash: str) -> dict[str, Any]:
    return {
        "id": item.id,
        "topic": item.topic,
        "source": item.source,
        "title": item.title,
        "content": item.content,
        "question": item.question,
        "input_hash": input_hash,
        "answer": "",
        "status": "error",
        "blocked": False,
        "reason_codes": [],
        "error": None,
        "error_type": None,
        "persona_model": persona_model,
        "api_style": None,
        "prompt_builder": "",
        "system_prompt_hash": None,
        "user_prompt_hash": None,
        "system_prompt": None,
        "user_prompt": None,
        "r1_count": 0,
        "r2_count": 0,
        "r3_count": 0,
        "rag_chunk_count": 0,
        "rag_score_mean": None,
        "latency_ms": 0,
        "attempts": 0,
    }


async def generate_one(
    *,
    analyzer: LeninAnalyzer,
    pipeline: AnalysisGenerationPipeline,
    item: QaItem,
    retries: int,
    save_full_prompts: bool,
) -> dict[str, Any]:
    input_hash = item.input_hash()
    row = base_row(item, persona_model=analyzer.generation_config.persona_model, input_hash=input_hash)
    key_concepts = analyzer.extract_key_concepts(item.content)
    enhanced_query = f"{item.title} {item.content[:200]} {' '.join(key_concepts)}"
    attempts = 0
    last_error: Exception | None = None
    max_tries = max(1, retries + 1)
    while attempts < max_tries:
        attempts += 1
        row["attempts"] = attempts
        try:
            result = await pipeline.generate(
                news_title=item.title,
                news_content=item.content,
                enhanced_query=enhanced_query,
                key_concepts=key_concepts,
                warn_only_guard=True,
            )
            answer = (result.guard_result.moderated_text or result.analysis or "").strip()
            row["latency_ms"] = int(result.latency_ms)
            row["api_style"] = result.metadata.get("api_style")
            row["prompt_builder"] = result.prompt_builder
            row["system_prompt_hash"] = sha256_text(result.system_prompt)
            row["user_prompt_hash"] = sha256_text(result.user_prompt)
            if save_full_prompts:
                row["system_prompt"] = result.system_prompt
                row["user_prompt"] = result.user_prompt
            for key in ("r1_count", "r2_count", "r3_count", "rag_chunk_count", "rag_score_mean"):
                row[key] = result.metadata.get(key, 0 if key != "rag_score_mean" else None)
            if not answer:
                row["status"] = "error"
                row["error"] = "empty model content"
                row["error_type"] = "empty_response"
                return row
            row["reason_codes"] = list(result.guard_result.reason_codes) + list(result.hallucination_codes)
            row["blocked"] = bool(result.guard_result.blocked)
            row["answer"] = answer
            row["status"] = "done"
            return row
        except Exception as error:  # noqa: BLE001
            last_error = error
            error_type = classify_error(error)
            row["error"] = str(error)[:500]
            row["error_type"] = error_type
            if error_type != "transient" or attempts >= max_tries:
                row["status"] = "error"
                return row
            logger.info("retry id=%s attempt=%s error_type=%s", item.id, attempts, error_type)
    row["status"] = "error"
    row["error"] = str(last_error)[:500] if last_error else "unknown"
    row["error_type"] = classify_error(last_error) if last_error else "runtime"
    return row

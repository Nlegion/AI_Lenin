"""DialecticalEngine: single-pass JSON reasoning with optional repair."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.core.analysis.evidence_brief import EvidenceBrief
from src.core.dialectics.config import DialecticalReasoningConfig
from src.core.dialectics.fact_span import entity_tokens_in_news, extract_fact_span
from src.core.dialectics.outcomes import (
    finalize_result,
    simplified_result,
    terminal_result,
)
from src.core.dialectics.packing import pack_reasoning_context
from src.core.dialectics.parse import parse_json_object
from src.core.dialectics.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.core.dialectics.rag_brief import build_principle_cards, cards_by_stance
from src.core.dialectics.repair import build_repair_request, error_set_progressed
from src.core.dialectics.schemas import DialecticalRequest, DialecticalResult
from src.core.dialectics.validators import build_result_from_payload, validate_result
from src.core.llm.base import GenerationBackend, GenerationRequest

logger = logging.getLogger(__name__)


class DialecticalEngine:
    def __init__(
        self,
        *,
        backend: GenerationBackend,
        config: DialecticalReasoningConfig,
    ) -> None:
        self.backend = backend
        self.config = config

    async def analyze(
        self,
        *,
        request: DialecticalRequest,
        brief: EvidenceBrief | None,
        enable_repair: bool = False,
    ) -> DialecticalResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        if brief is None and not request.fixture_mode:
            return terminal_result(
                outcome="suppress",
                reason_codes=["missing_brief"],
                timings=timings,
                started=started,
            )

        t0 = time.perf_counter()
        fact = extract_fact_span(
            news_title=request.news_title,
            news_content=request.news_content,
        )
        news_blob = f"{request.news_title}\n{request.news_content}"
        if not entity_tokens_in_news(fact, news_blob):
            fact = request.news_title.strip() or fact
        timings["fact_span_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        cards = (
            build_principle_cards(brief, config=self.config)
            if brief is not None
            else []
        )
        by_stance = cards_by_stance(cards)
        has_r1 = bool(by_stance.get("core_self"))
        has_r3 = bool(by_stance.get("influence_critical"))
        timings["principle_cards_ms"] = (time.perf_counter() - t0) * 1000.0

        reason_codes: list[str] = []
        if not has_r3:
            reason_codes.append("r3_absent")
        if not has_r1 or not request.dialectical_applicable:
            if not has_r1:
                reason_codes.append("insufficient_evidence")
            if not request.dialectical_applicable:
                reason_codes.append("dialectical_not_applicable")
            return simplified_result(
                fact=fact,
                reason_codes=reason_codes,
                timings=timings,
                started=started,
                cards=cards,
                config=self.config,
            )

        packed = pack_reasoning_context(
            news_title=request.news_title,
            news_fact=fact,
            cards=cards,
            system_prompt=SYSTEM_PROMPT,
            user_prefix=USER_TEMPLATE,
            config=self.config,
        )
        user = USER_TEMPLATE.format(
            news_block=packed.news_block,
            principle_block=packed.principle_block,
        )
        gen_request = GenerationRequest(
            system_prompt=SYSTEM_PROMPT,
            user_content=user,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )

        t0 = time.perf_counter()
        try:
            response = await self.backend.generate(request=gen_request)
        except Exception as exc:  # noqa: BLE001
            logger.exception("dialectical_backend_error")
            return terminal_result(
                outcome="suppress",
                reason_codes=[*reason_codes, "backend_error"],
                timings=timings,
                started=started,
                metadata={"error": str(exc)[:200]},
            )
        timings["reasoning_ms"] = (time.perf_counter() - t0) * 1000.0
        if response.finish_reason == "length":
            reason_codes.append("length_truncated")

        parsed = parse_json_object(response.text)
        if parsed.status == "fail" or parsed.data is None:
            if enable_repair:
                repaired = await self._repair_loop(
                    packed=packed,
                    previous_payload={"raw": response.text[:1500]},
                    errors=["json_parse_failed"],
                    cards=cards,
                    has_r3=has_r3,
                    timings=timings,
                )
                if repaired is not None:
                    return finalize_result(
                        result=repaired,
                        reason_codes=reason_codes,
                        timings=timings,
                        started=started,
                        has_r3=has_r3,
                        config=self.config,
                    )
            return terminal_result(
                outcome="hold_review",
                reason_codes=[*reason_codes, "parse_error"],
                timings=timings,
                started=started,
                metadata={"parse_status": parsed.status},
            )

        payload: dict[str, Any] = parsed.data
        result = build_result_from_payload(payload=payload, cards=cards)
        result.quality = validate_result(result=result, cards=cards, has_r3=has_r3)
        if not result.quality.passed and enable_repair:
            repaired = await self._repair_loop(
                packed=packed,
                previous_payload=payload,
                errors=list(result.quality.errors),
                cards=cards,
                has_r3=has_r3,
                timings=timings,
            )
            if repaired is not None:
                result = repaired
        return finalize_result(
            result=result,
            reason_codes=reason_codes,
            timings=timings,
            started=started,
            has_r3=has_r3,
            config=self.config,
        )

    async def _repair_loop(
        self,
        *,
        packed,
        previous_payload: dict[str, Any],
        errors: list[str],
        cards,
        has_r3: bool,
        timings: dict[str, float],
    ) -> DialecticalResult | None:
        best: DialecticalResult | None = None
        prev_errors = list(errors)
        payload = previous_payload
        for attempt in range(1, self.config.repair_max_attempts + 1):
            t0 = time.perf_counter()
            req = build_repair_request(
                packed=packed,
                previous_payload=payload,
                errors=prev_errors,
                config=self.config,
            )
            try:
                response = await self.backend.generate(request=req)
            except Exception:  # noqa: BLE001
                logger.exception("dialectical_repair_backend_error attempt=%s", attempt)
                break
            timings[f"repair_{attempt}_ms"] = (time.perf_counter() - t0) * 1000.0
            if response.finish_reason == "length":
                break
            parsed = parse_json_object(response.text)
            if parsed.data is None:
                continue
            payload = parsed.data
            candidate = build_result_from_payload(payload=payload, cards=cards)
            candidate.quality = validate_result(
                result=candidate, cards=cards, has_r3=has_r3
            )
            candidate.metadata["repair_attempt"] = attempt
            best = candidate
            if candidate.quality.passed:
                return candidate
            if not error_set_progressed(prev_errors, candidate.quality.errors):
                candidate.reason_codes = [*candidate.reason_codes, "repair_same_error"]
                break
            prev_errors = list(candidate.quality.errors)
        return best

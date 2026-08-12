"""Standalone pre-RAG censorship wrapper with normalized decision contract."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from src.core.safety.news_guard import NewsGuard
from src.core.safety.censor_hashing import NORMALIZER_VERSION, canonical_json_hash, compute_content_hash
from src.core.safety.semantic_l2_classifier import SemanticL2Classifier
from src.core.safety.safety_gate import SafetyGate
from src.core.safety.safety_gate_types import GateContext, SafetyHint
from src.core.safety.pre_rag_censor_types import CensorDecision, CensorInput, CensorResult, NormalizationMeta

logger = logging.getLogger(__name__)

_RU_CHAR_RE = re.compile(r"[а-яёА-ЯЁ]")
_ALPHA_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
_SPORT_TOKEN_RE = re.compile(
    r"\b(спорт|футбол|хоккей|теннис|волейбол|баскетбол|матч|турнир|чемпионат|"
    r"атлет|спортсмен|тренер|федерац|лига|кубок|олимп|паралимп|гто)\w*\b",
    re.IGNORECASE,
)
_SPORT_TEAM_RE = re.compile(
    r"\b(спартак|цска|зенит|локомотив|динамо|торпедо|ростов|рубин|крылья\s+советов|"
    r"ахмат|авангард|ак\s*барс|ска|трактор|металлург|салават\s+юлаев|"
    r"первая\s+лига|премьер-?лига|рпл|кхл|нхл)\b",
    re.IGNORECASE,
)
_AIRPORT_TEMPLATE_RE = re.compile(
    r"(аэропорт\w*).{0,60}(временн\w*\s+ограничени\w*|ограничени\w*|возобновил\w*\s+работ)",
    re.IGNORECASE,
)
_SPECULATIVE_TERMS = (
    "оценил",
    "шансы",
    "прогноз",
    "вероятность",
    "ожидается",
    "может",
    "угроза",
    "ужесточение",
    "обсудил",
    "предположил",
)
_CRISIS_TERMS = (
    "атака",
    "бпла",
    "дрон",
    "взрыв",
    "хлопок",
    "пожар",
    "чп",
    "авария",
    "пострадав",
    "погиб",
    "эвакуац",
    "опасност",
    "закрыт",
    "рейс",
    "угроза",
    "тревог",
)
_MANUAL_WAR_TERMS = (
    "бпла",
    "взрыв",
    "беспилотн",
    "авиационн",
    "опасност",
    "бойцы",
    "аэропорт приостановил",
    "дрон",
    "всу",
)
_MANUAL_TERRACT_TERMS = ("wildberries",)
_MANUAL_FIRE_TERMS = ("пожар", "пожары")
_MANUAL_AIRPORT_TERMS = ("аэропорт",)
_MANUAL_RELIGION_TERMS = ("храм",)
_MANUAL_DEATH_TERMS = ("останк", "труп")
_ETHNO_HATE_HARD_TERMS = (
    "русопет",
    "чурк",
    "хач",
    "черножоп",
    "узкоглаз",
    "инородц",
    "нацмен",
    "малоросс",
)
_ETHNO_HATE_ACTION_TERMS = (
    "ненав",
    "убива",
    "изгна",
    "депорт",
    "очист",
    "запрет",
    "уничтож",
)
_SEPARATOR_RE = re.compile(r"[\s\-\._:,;!?/\\|()\[\]{}\"'`~]+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\uFEFF]")
_MANUAL_WAR_GENERIC_TERMS = (
    "бомб",
    "война",
    "вторая мировая",
    "великой отечественной",
    "великая отечественная",
    "нацист",
    "битв",
    "сражен",
    "фронт",
    "боестолкнов",
    "наступлен",
    "контрнаступ",
    "штурм",
)
_MANUAL_SPORT_TERMS = (
    "гимнаст",
    "фигурист",
    "isu",
    "роднин",
    "спартак",
    "цска",
    "зенит",
    "локомотив",
    "динамо",
    "первая лига",
    "рпл",
    "кхл",
)
_CATEGORY_ALIASES = {
    "airport": "AIRPORT",
    "religion": "RELIGION",
    "death": "DEATH",
    "war": "WAR",
    "fire": "FIRE",
    "теракт": "TERRACT",
    "терракт": "TERRACT",
}

_DEFAULT_L2_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "DIPLOMACY": ("дипломатия", "переговоры", "посол", "международные отношения"),
    "SANCTIONS": ("санкции", "ограничения", "экспорт", "импорт", "торговля"),
    "MILITARY_OFFICIAL_STATEMENT": ("минобороны", "заявление", "брифинг", "официально"),
    "PROTESTS": ("митинг", "протест", "демонстрация", "забастовка"),
    "ETHNIC_RELIGIOUS": ("национальн", "этническ", "религиозн"),
    "SENSITIVE_POLITICS": ("власть", "оппозиция", "политическ"),
}


@dataclass
class CensorRuntimeConfig:
    min_chars: int = 24
    ru_ratio_threshold: float = 0.60
    non_ru_mode: str = "skip_non_ru"  # skip_non_ru|review_non_ru
    duplicate_ttl_seconds: int = 24 * 60 * 60
    duplicate_cache_size: int = 100_000
    sport_block_enabled: bool = True
    l2_similarity_enabled: bool = True
    l2_review_threshold: float = 0.72
    l2_hard_block_threshold: float = 0.92
    l2_latency_budget_ms: float = 20.0
    l2_text_max_chars: int = 1200
    l2_cache_size: int = 4096
    l3_enabled: bool = False
    l3_timeout_seconds: float = 2.0
    l3_retry_count: int = 1
    l3_circuit_open_seconds: float = 120.0
    fallback_to_review: bool = True
    hot_reload_enabled: bool = True
    hot_reload_poll_seconds: float = 60.0
    unknown_topic_to_skip_enabled: bool = True
    unknown_low_signal_l2_max: float = 0.25
    airport_operational_whitelist_enabled: bool = True
    sanctions_allow_l2_min: float = 0.60
    require_l3_for_sanctions_allow: bool = False
    sensitive_topic_guard_enabled: bool = True
    ethno_hate_containment_enabled: bool = True
    war_review_threshold: float = 0.50
    war_hard_block_threshold: float = 0.80
    l2_model_version: str = "l2-default"
    cache_cleanup_interval_seconds: int = 3600
    cache_ttl_seconds: int = 24 * 60 * 60


def compose_decision(
    *,
    l1_decision: CensorDecision,
    l2_signal: tuple[CensorDecision, float] | None,
    l3_decision: CensorDecision | None,
) -> CensorDecision:
    """Decision composition truth table: L1 > L3 > L2 for stage 1."""
    if l1_decision in {"hard_block", "skip"}:
        return l1_decision
    if l3_decision is not None:
        return l3_decision
    if l2_signal is None:
        return l1_decision
    l2_decision, _score = l2_signal
    if l1_decision == "review":
        return l2_decision if l2_decision in {"review", "hard_block"} else "review"
    if l1_decision == "allow":
        return l2_decision if l2_decision in {"allow", "review", "hard_block"} else "allow"
    return l1_decision


def _normalize_category(category: str | None) -> str | None:
    if not category:
        return None
    lowered = category.strip().lower()
    if lowered in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[lowered]
    return category.strip().upper()


class PreRagCensor:
    """Centralized censorship module called before RAG/generation."""

    def __init__(
        self,
        *,
        safety_gate: SafetyGate | None,
        news_guard: NewsGuard | None,
        config: CensorRuntimeConfig | None = None,
        l3_reviewer: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
        config_path: str | None = None,
        load_cached_decision: Callable[[str, str], Awaitable[dict[str, Any] | None]] | None = None,
        save_cached_decision: Callable[[str, str, str, CensorResult], Awaitable[None]] | None = None,
        cleanup_cached_decisions: Callable[[int], Awaitable[int]] | None = None,
    ):
        self._safety_gate = safety_gate
        self._news_guard = news_guard
        self._config = config or CensorRuntimeConfig()
        self._l3_reviewer = l3_reviewer
        self._config_path = config_path
        self._config_path_hash = ""
        self._policy_hash = ""
        self._model_version_hash = ""
        self._config_version_hash = ""
        self._last_reload_check = 0.0
        self._last_cache_cleanup = 0.0
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._hash_locks: dict[str, asyncio.Lock] = {}
        self._l2_disabled_until = 0.0
        self._l3_circuit_open_until = 0.0
        self._load_cached_decision = load_cached_decision
        self._save_cached_decision = save_cached_decision
        self._cleanup_cached_decisions = cleanup_cached_decisions
        self._l2_classifier = SemanticL2Classifier(
            model_version=self._config.l2_model_version,
            prototypes=_DEFAULT_L2_PROTOTYPES,
            text_max_chars=self._config.l2_text_max_chars,
            cache_size=self._config.l2_cache_size,
        )
        self._refresh_version_hashes()

    @property
    def config_version_hash(self) -> str:
        return self._config_version_hash

    @property
    def model_version_hash(self) -> str:
        return self._model_version_hash

    def _refresh_version_hashes(self) -> None:
        runtime_payload = {
            "min_chars": self._config.min_chars,
            "ru_ratio_threshold": self._config.ru_ratio_threshold,
            "non_ru_mode": self._config.non_ru_mode,
            "sport_block_enabled": self._config.sport_block_enabled,
            "l2_similarity_enabled": self._config.l2_similarity_enabled,
            "l2_review_threshold": self._config.l2_review_threshold,
            "l2_hard_block_threshold": self._config.l2_hard_block_threshold,
            "l2_text_max_chars": self._config.l2_text_max_chars,
            "l2_cache_size": self._config.l2_cache_size,
            "unknown_topic_to_skip_enabled": self._config.unknown_topic_to_skip_enabled,
            "unknown_low_signal_l2_max": self._config.unknown_low_signal_l2_max,
            "sanctions_allow_l2_min": self._config.sanctions_allow_l2_min,
            "require_l3_for_sanctions_allow": self._config.require_l3_for_sanctions_allow,
            "sensitive_topic_guard_enabled": self._config.sensitive_topic_guard_enabled,
            "ethno_hate_containment_enabled": self._config.ethno_hate_containment_enabled,
            "war_review_threshold": self._config.war_review_threshold,
            "war_hard_block_threshold": self._config.war_hard_block_threshold,
        }
        self._policy_hash = canonical_json_hash(runtime_payload)[:16]
        self._model_version_hash = (
            "l2_off" if not self._config.l2_similarity_enabled else self._config.l2_model_version
        )
        self._l2_classifier = SemanticL2Classifier(
            model_version=self._model_version_hash,
            prototypes=_DEFAULT_L2_PROTOTYPES,
            text_max_chars=self._config.l2_text_max_chars,
            cache_size=self._config.l2_cache_size,
        )
        self._config_version_hash = canonical_json_hash(
            {
                "normalizer_version": NORMALIZER_VERSION,
                "policy_hash": self._policy_hash,
                "model_version_hash": self._model_version_hash,
            }
        )[:16]

    async def _maybe_cleanup_cache(self) -> None:
        if self._cleanup_cached_decisions is None:
            return
        now = time.time()
        if now - self._last_cache_cleanup < max(self._config.cache_cleanup_interval_seconds, 60):
            return
        self._last_cache_cleanup = now
        try:
            await self._cleanup_cached_decisions(self._config.cache_ttl_seconds)
        except Exception as error:  # noqa: BLE001
            logger.warning("censor_cache_cleanup_failed err=%s", error)

    async def evaluate(self, payload: CensorInput) -> CensorResult:
        self._maybe_reload_runtime_config()
        await self._maybe_cleanup_cache()
        started = time.perf_counter()
        normalization = self._normalize(payload=payload)
        lock = self._hash_locks.setdefault(normalization.content_hash, asyncio.Lock())
        async with lock:
            cached = await self._read_cached_result(normalization=normalization, started=started)
            if cached is not None:
                return cached

            result = await self._evaluate_uncached(payload=payload, normalization=normalization, started=started)
            await self._write_cached_result(normalization=normalization, result=result)
            return result

    async def _evaluate_uncached(
        self,
        *,
        payload: CensorInput,
        normalization: NormalizationMeta,
        started: float,
    ) -> CensorResult:
        cfg = self._config
        manual_override = self._manual_hard_block_override(normalization.normalized_text)
        if manual_override is not None:
            decision, category, code, reason = manual_override
            return self._build_result(
                decision=decision,
                category=category,
                reason_codes=[code],
                reason=reason,
                normalization=normalization,
                started=started,
            )
        l0_result = self._l0_decision(normalization=normalization, cfg=cfg)
        if l0_result is not None:
            l0_decision, l0_code = l0_result
            return self._build_result(
                decision=l0_decision,
                category="NON_TOPICAL",
                reason_codes=["l0_filtered", l0_code],
                reason="L0 technical filter",
                normalization=normalization,
                started=started,
            )

        full_text = f"{payload.title}\n{payload.body}".lower()
        if (
            cfg.airport_operational_whitelist_enabled
            and self._is_airport_operational(full_text)
            and not self._has_crisis_keywords(full_text)
            and self._is_trusted_source(payload.source)
        ):
            return self._build_result(
                decision="skip",
                category="NON_TOPICAL",
                reason_codes=["airport_operational_whitelist"],
                reason="Operational airport update without crisis markers",
                normalization=normalization,
                started=started,
            )

        if cfg.sport_block_enabled and self._is_sport(normalization.normalized_text):
            return self._build_result(
                decision="hard_block",
                category="SPORT_BLOCKED",
                reason_codes=["sport_blocked"],
                reason="Sport is blocked by policy",
                normalization=normalization,
                started=started,
            )

        try:
            l1_result = self._evaluate_l1(payload=payload, started=started)
        except Exception as error:  # noqa: BLE001
            logger.exception("l1_evaluation_failed err=%s", error)
            war_score = self._war_signal_score(normalization.normalized_text)
            fallback_decision: CensorDecision = (
                "hard_block" if war_score >= cfg.war_hard_block_threshold else "review"
            )
            return self._build_result(
                decision=fallback_decision,
                category="WAR_OPERATIONAL" if fallback_decision == "hard_block" else "SENSITIVE_POLITICS",
                reason_codes=["l1_error_fallback"],
                reason="L1 evaluation failed",
                normalization=normalization,
                started=started,
            )
        try:
            l2_signal = self._evaluate_l2(normalization.normalized_text, cfg=cfg)
        except Exception as error:  # noqa: BLE001
            logger.warning("l2_evaluation_failed err=%s", error)
            l2_signal = None
            extra_l2_error_code = "l2_error_fallback"
        else:
            extra_l2_error_code = None
        l3_decision = await self._evaluate_l3(payload=payload, base=l1_result.decision, cfg=cfg)
        final_decision = compose_decision(
            l1_decision=l1_result.decision,
            l2_signal=l2_signal,
            l3_decision=l3_decision,
        )
        l2_score = float(l2_signal[1]) if l2_signal is not None else None
        l3_used = l3_decision is not None
        base_reason_codes = list(l1_result.reason_codes)
        if extra_l2_error_code is not None:
            base_reason_codes.append(extra_l2_error_code)
        final_decision, final_category, final_codes = self._apply_policy_overrides(
            cfg=cfg,
            decision=final_decision,
            category=l1_result.category,
            reason_codes=base_reason_codes,
            l2_score=l2_score,
            l3_used=l3_used,
            text_lower=full_text,
            source=payload.source,
        )

        confidence = dict(l1_result.confidence)
        if l2_signal is not None:
            confidence["l2_similarity"] = float(l2_signal[1])
        if l3_decision is not None:
            confidence["l3_used"] = 1.0
        result = CensorResult(
            decision=final_decision,
            category=_normalize_category(final_category),
            risk_tier=self._tier_for_decision(final_decision, l1_result.risk_tier),
            reason_codes=list(dict.fromkeys(final_codes)),
            reason=l1_result.reason,
            message=l1_result.message,
            confidence=confidence,
            context_hints=list(l1_result.context_hints),
            needs_yellow_warning=l1_result.needs_yellow_warning,
            audit={
                **l1_result.audit,
                "l1_decision": l1_result.decision,
                "normalization": {
                    "ru_ratio": normalization.ru_ratio,
                    "duplicate_hit": normalization.duplicate_hit,
                    "duplicate_age_seconds": normalization.duplicate_age_seconds,
                },
                "l2_signal": l2_signal,
                "l3_decision": l3_decision,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "runtime_config_hash": self._config_path_hash,
                "normalizer_version": NORMALIZER_VERSION,
                "model_version_hash": self._model_version_hash,
                "config_version_hash": self._config_version_hash,
            },
        )
        return result

    async def _read_cached_result(
        self,
        *,
        normalization: NormalizationMeta,
        started: float,
    ) -> CensorResult | None:
        if self._load_cached_decision is None:
            return None
        cached = await self._load_cached_decision(normalization.content_hash, self._config_version_hash)
        if cached is None:
            return None
        raw_hints = list(cached.get("context_hints") or [])
        hints: list[SafetyHint] = []
        for item in raw_hints:
            if isinstance(item, SafetyHint):
                hints.append(item)
                continue
            try:
                hints.append(SafetyHint(str(item)))
            except ValueError:
                continue
        return CensorResult(
            decision=cached["decision"],
            category=_normalize_category(cached.get("category")),
            risk_tier=cached["risk_tier"],
            reason_codes=list(cached["reason_codes"]),
            reason="Duplicate decision cache hit",
            confidence=dict(cached.get("confidence") or {}),
            context_hints=hints,
            needs_yellow_warning=bool(cached.get("needs_yellow_warning", False)),
            audit={
                "cache_hit": True,
                "normalizer_version": NORMALIZER_VERSION,
                "config_version_hash": self._config_version_hash,
                "model_version_hash": self._model_version_hash,
                "normalization": {
                    "ru_ratio": normalization.ru_ratio,
                    "duplicate_hit": normalization.duplicate_hit,
                    "duplicate_age_seconds": normalization.duplicate_age_seconds,
                },
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            },
        )

    async def _write_cached_result(
        self,
        *,
        normalization: NormalizationMeta,
        result: CensorResult,
    ) -> None:
        if self._save_cached_decision is None:
            return
        try:
            await self._save_cached_decision(
                normalization.content_hash,
                self._config_version_hash,
                self._model_version_hash,
                result,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("censor_cache_save_failed err=%s", error)

    def _normalize(self, *, payload: CensorInput) -> NormalizationMeta:
        url = str(payload.metadata.get("url", "")) if payload.metadata else ""
        content_hash, normalized = compute_content_hash(title=payload.title, body=payload.body, url=url)
        ru_chars = len(_RU_CHAR_RE.findall(normalized))
        alpha_chars = len(_ALPHA_RE.findall(normalized))
        ru_ratio = float(ru_chars / alpha_chars) if alpha_chars else 0.0
        duplicate_hit, duplicate_age = self._mark_and_check_duplicate(content_hash=content_hash)
        return NormalizationMeta(
            content_hash=content_hash,
            normalized_text=normalized,
            normalizer_version=NORMALIZER_VERSION,
            ru_ratio=ru_ratio,
            empty=len(normalized) < self._config.min_chars,
            duplicate_hit=duplicate_hit,
            duplicate_age_seconds=duplicate_age,
        )

    def _l0_decision(
        self,
        *,
        normalization: NormalizationMeta,
        cfg: CensorRuntimeConfig,
    ) -> tuple[CensorDecision, str] | None:
        if normalization.empty:
            return ("skip", "l0_too_short")
        if normalization.ru_ratio < cfg.ru_ratio_threshold:
            return (
                ("review", "l0_non_ru_review")
                if cfg.non_ru_mode == "review_non_ru"
                else ("skip", "l0_non_ru_skip")
            )
        return None

    def _evaluate_l1(self, *, payload: CensorInput, started: float) -> CensorResult:
        if self._safety_gate is not None and self._safety_gate.config.flags.enabled:
            ctx = GateContext(
                title=payload.title,
                content=payload.body,
                source=payload.source,
                item_id=payload.news_id,
                config_version_hash=self._safety_gate.config_version_hash,
            )
            compare = self._safety_gate.evaluate_with_shadow(ctx, legacy_guard=self._news_guard)
            base = compare.enforced
            legacy_decision = str(base.decision)
            mapped = self._map_legacy_decision(legacy_decision)
            category = self._infer_category(reason_codes=base.reason_codes, text=f"{payload.title}\n{payload.body}")
            return CensorResult(
                decision=mapped,
                category=category,
                risk_tier=self._tier_for_decision(mapped, str(base.risk_tier)),
                reason_codes=list(base.reason_codes),
                reason=base.reason,
                message=base.message,
                confidence={"l1_rule": 1.0},
                context_hints=list(base.context_hints),
                needs_yellow_warning=base.needs_yellow_warning,
                audit={
                    "legacy_decision": legacy_decision,
                    "legacy_mapping": mapped,
                    "shadow": {
                        "decision_match": compare.decision_match,
                        "old_decision": compare.old_decision.decision if compare.old_decision else None,
                        "new_decision": compare.new_decision.decision if compare.new_decision else None,
                        "reason_diff": list(compare.reason_diff),
                    },
                    "config_version_hash": compare.config_version_hash,
                    "l1_latency_ms": (time.perf_counter() - started) * 1000.0,
                },
            )

        if self._news_guard is not None:
            base = self._news_guard.evaluate_input(payload.title, payload.body, source=payload.source)
            mapped = self._map_legacy_decision(str(base.decision))
            category = self._infer_category(reason_codes=base.reason_codes, text=f"{payload.title}\n{payload.body}")
            hints = []
            if base.risk_tier == "yellow":
                hints = [SafetyHint.YELLOW_CONSTRAINED_ANALYSIS, SafetyHint.AVOID_COMBAT_ESTIMATES]
            return CensorResult(
                decision=mapped,
                category=category,
                risk_tier=self._tier_for_decision(mapped, str(base.risk_tier)),
                reason_codes=list(base.reason_codes),
                reason=base.reason,
                message=base.message,
                confidence={"l1_rule": 1.0},
                context_hints=hints,
                needs_yellow_warning=base.risk_tier == "yellow" and mapped == "allow",
                audit={"legacy_decision": base.decision, "legacy_mapping": mapped},
            )

        return self._build_result(
            decision="review",
            category=None,
            reason_codes=["no_gate_available"],
            reason="No censorship backend available",
            normalization=NormalizationMeta("", "", NORMALIZER_VERSION, 0.0, False, False, None),
            started=started,
        )

    def _evaluate_l2(self, text: str, *, cfg: CensorRuntimeConfig) -> tuple[CensorDecision, float] | None:
        if not cfg.l2_similarity_enabled:
            return None
        if time.time() < self._l2_disabled_until:
            return None
        started = time.perf_counter()
        l2_score = self._l2_classifier.score(text)
        category, score = l2_score.category, l2_score.score
        war_score = self._war_signal_score(text)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > cfg.l2_latency_budget_ms:
            self._l2_disabled_until = time.time() + 60.0
            logger.warning("l2_similarity_auto_disabled elapsed_ms=%.2f", elapsed_ms)
            return None
        if war_score >= cfg.war_hard_block_threshold:
            return ("hard_block", max(score, war_score))
        if war_score >= cfg.war_review_threshold:
            return ("review", max(score, war_score))
        if score >= cfg.l2_hard_block_threshold and category not in {"DIPLOMACY", "SANCTIONS"}:
            return ("hard_block", score)
        if score >= cfg.l2_review_threshold:
            return ("review", score)
        return ("allow", score)

    async def _evaluate_l3(
        self,
        *,
        payload: CensorInput,
        base: CensorDecision,
        cfg: CensorRuntimeConfig,
    ) -> CensorDecision | None:
        if base != "review" or not cfg.l3_enabled or self._l3_reviewer is None:
            return None
        if time.time() < self._l3_circuit_open_until:
            return "review"
        attempts = max(cfg.l3_retry_count, 0) + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                raw = await asyncio.wait_for(
                    self._l3_reviewer(payload.title, payload.body),
                    timeout=cfg.l3_timeout_seconds,
                )
                decision = str(raw.get("decision", "review"))
                if decision in {"allow", "review", "hard_block"}:
                    return decision  # type: ignore[return-value]
                return "review"
            except Exception as error:  # noqa: BLE001
                last_error = error
                await asyncio.sleep(1.0)
        logger.warning("l3_review_failed err=%s", last_error)
        self._l3_circuit_open_until = time.time() + cfg.l3_circuit_open_seconds
        return "review" if cfg.fallback_to_review else "allow"

    def _map_legacy_decision(self, decision: str) -> CensorDecision:
        mapping: dict[str, CensorDecision] = {
            "allow": "allow",
            "deny": "hard_block",
            "quarantine": "review",
            "skip": "skip",
        }
        return mapping.get(decision, "review")

    def _tier_for_decision(
        self,
        decision: CensorDecision,
        legacy_risk_tier: str | None = None,
    ) -> str:
        if decision == "hard_block":
            return "red"
        if decision == "review":
            return "yellow"
        if decision == "skip":
            return "green"
        return legacy_risk_tier or "green"

    def _infer_category(self, *, reason_codes: list[str], text: str) -> str | None:
        lowered_codes = [code.lower() for code in reason_codes]
        joined = " ".join(lowered_codes)
        lower = text.lower()
        if any("sport" in code or "out_of_scope:sport" in code for code in lowered_codes):
            return "SPORT_BLOCKED"
        if any(token in joined for token in ("drone", "combat", "military", "сво", "обстрел")):
            return "WAR_OPERATIONAL"
        if any(token in joined for token in ("pii", "private_victim", "фио", "тел")):
            return "PERSONAL_DATA"
        if "sanctions" in lower or "санкц" in lower:
            return "SANCTIONS"
        if "диплом" in lower or "посол" in lower:
            return "DIPLOMACY"
        if "protest" in lower or "митинг" in lower:
            return "PROTESTS"
        if "national" in joined or "религи" in lower or "этничес" in lower:
            return "ETHNIC_RELIGIOUS"
        return None

    def _is_sport(self, text: str) -> bool:
        return bool(_SPORT_TOKEN_RE.search(text) or _SPORT_TEAM_RE.search(text))

    def _mark_and_check_duplicate(self, *, content_hash: str) -> tuple[bool, float | None]:
        now = time.time()
        existing = self._seen.get(content_hash)
        duplicate_hit = False
        duplicate_age: float | None = None
        if existing is not None and now - existing <= self._config.duplicate_ttl_seconds:
            duplicate_hit = True
            duplicate_age = now - existing
        self._seen[content_hash] = now
        self._seen.move_to_end(content_hash)
        cutoff = now - self._config.duplicate_ttl_seconds
        while self._seen:
            key, ts = next(iter(self._seen.items()))
            if ts >= cutoff and len(self._seen) <= self._config.duplicate_cache_size:
                break
            self._seen.popitem(last=False)
        return duplicate_hit, duplicate_age

    def _build_result(
        self,
        *,
        decision: CensorDecision,
        category: str | None,
        reason_codes: list[str],
        reason: str,
        normalization: NormalizationMeta,
        started: float,
    ) -> CensorResult:
        return CensorResult(
            decision=decision,
            category=_normalize_category(category),
            risk_tier=self._tier_for_decision(decision, None),
            reason_codes=list(reason_codes),
            reason=reason,
            confidence={"l1_rule": 1.0},
            context_hints=[],
            needs_yellow_warning=False,
            audit={
                "config_version_hash": self._config_version_hash,
                "model_version_hash": self._model_version_hash,
                "normalizer_version": NORMALIZER_VERSION,
                "normalization": {
                    "ru_ratio": normalization.ru_ratio,
                    "duplicate_hit": normalization.duplicate_hit,
                    "duplicate_age_seconds": normalization.duplicate_age_seconds,
                },
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "runtime_config_hash": self._config_path_hash,
            },
            timestamp_utc=datetime.now(timezone.utc),
        )

    def _is_trusted_source(self, source: str | None) -> bool:
        return str(source or "").strip().upper() == "TASS"

    def _is_airport_operational(self, text_lower: str) -> bool:
        return bool(_AIRPORT_TEMPLATE_RE.search(text_lower))

    def _has_crisis_keywords(self, text_lower: str) -> bool:
        return any(token in text_lower for token in _CRISIS_TERMS)

    def _manual_hard_block_override(
        self,
        text_lower: str,
    ) -> tuple[CensorDecision, str, str, str] | None:
        if self._config.ethno_hate_containment_enabled and self._has_ethno_hate_markers(text_lower):
            return (
                "hard_block",
                "ETHNIC_RELIGIOUS",
                "manual_ethno_hate_containment",
                "Manual rule: ethno-hate containment hard block",
            )
        if "воздушн" in text_lower and "тревог" in text_lower:
            return (
                "hard_block",
                "WAR_OPERATIONAL",
                "manual_war_operational_hard_block",
                "Manual rule: air alert hard block",
            )
        if any(token in text_lower for token in _MANUAL_WAR_TERMS):
            return (
                "hard_block",
                "WAR_OPERATIONAL",
                "manual_war_operational_hard_block",
                "Manual rule: war-related keyword hard block",
            )
        if any(token in text_lower for token in _MANUAL_SPORT_TERMS):
            return (
                "hard_block",
                "SPORT_BLOCKED",
                "manual_sport_hard_block",
                "Manual rule: sport hard block",
            )
        if any(token in text_lower for token in _MANUAL_AIRPORT_TERMS):
            return ("hard_block", "AIRPORT", "manual_airport_hard_block", "Manual rule: airport hard block")
        if any(token in text_lower for token in _MANUAL_RELIGION_TERMS):
            return ("hard_block", "RELIGION", "manual_religion_hard_block", "Manual rule: religion hard block")
        if any(token in text_lower for token in _MANUAL_DEATH_TERMS):
            return ("hard_block", "DEATH", "manual_death_hard_block", "Manual rule: death hard block")
        if any(token in text_lower for token in _MANUAL_WAR_GENERIC_TERMS):
            return ("hard_block", "WAR", "manual_war_hard_block", "Manual rule: war hard block")
        if any(token in text_lower for token in _MANUAL_TERRACT_TERMS):
            return ("hard_block", "TERRACT", "manual_wildberries_terract", "Manual rule: Wildberries hard block")
        if any(token in text_lower for token in _MANUAL_FIRE_TERMS):
            return ("hard_block", "FIRE", "manual_fire_hard_block", "Manual rule: fire hard block")
        return None

    def _has_ethno_hate_markers(self, text_lower: str) -> bool:
        cleaned = _ZERO_WIDTH_RE.sub("", text_lower)
        compact = _SEPARATOR_RE.sub("", cleaned)
        hard_hit = any(term in cleaned or term in compact for term in _ETHNO_HATE_HARD_TERMS)
        if not hard_hit:
            return False
        action_hit = any(term in cleaned or term in compact for term in _ETHNO_HATE_ACTION_TERMS)
        # Keep containment strict but avoid broad false positives for neutral mentions.
        return action_hit or any(term in cleaned for term in ("нужно", "должны", "пора", "против"))

    def _war_signal_score(self, text_lower: str) -> float:
        terms = (
            "сво",
            "всу",
            "бпла",
            "дрон",
            "ракет",
            "обстрел",
            "удар",
            "пво",
            "минобороны",
            "боев",
            "погиб",
            "битв",
            "сражен",
            "фронт",
            "боестолкнов",
            "наступлен",
            "контрнаступ",
            "штурм",
        )
        hits = sum(1 for term in terms if term in text_lower)
        return min(hits / 5.0, 1.0)

    def _apply_policy_overrides(
        self,
        *,
        cfg: CensorRuntimeConfig,
        decision: CensorDecision,
        category: str | None,
        reason_codes: list[str],
        l2_score: float | None,
        l3_used: bool,
        text_lower: str,
        source: str | None,
    ) -> tuple[CensorDecision, str | None, list[str]]:
        codes = list(reason_codes)
        current_decision = decision
        current_category = category
        sensitive_topics = {"SANCTIONS", "DIPLOMACY"}
        hard_block_overrides = {"sport_blocked", "drone", "combat", "terror", "violence"}
        war_score = self._war_signal_score(text_lower)

        if (
            cfg.sensitive_topic_guard_enabled
            and current_decision == "hard_block"
            and current_category in sensitive_topics
            and not any(any(marker in code.lower() for marker in hard_block_overrides) for code in codes)
        ):
            current_decision = "review"
            codes.append("sensitive_topic_guard")

        if current_category == "SANCTIONS" and current_decision == "allow":
            speculative = any(token in text_lower for token in _SPECULATIVE_TERMS)
            low_confidence = l2_score is None or l2_score < cfg.sanctions_allow_l2_min
            if low_confidence or speculative or not self._is_trusted_source(source) or (
                cfg.require_l3_for_sanctions_allow and not l3_used
            ):
                current_decision = "review"
                codes.append("sanctions_allow_gate")

        if (
            cfg.unknown_topic_to_skip_enabled
            and current_decision == "review"
            and codes == ["unknown_topic"]
            and (l2_score is None or l2_score < cfg.unknown_low_signal_l2_max)
            and self._is_trusted_source(source)
            and not self._has_crisis_keywords(text_lower)
        ):
            current_decision = "allow"
            current_category = "NON_TOPICAL"
            codes = [
                "unknown_topic_low_signal_allow_forward",
                "override:unknown_topic_forward_trusted_source",
            ]

        if war_score >= cfg.war_hard_block_threshold and current_decision != "hard_block":
            current_decision = "hard_block"
            current_category = "WAR_OPERATIONAL"
            codes.append("war_signal_hard_block")
        elif war_score >= cfg.war_review_threshold and current_decision == "allow":
            current_decision = "review"
            current_category = current_category or "MILITARY_OFFICIAL_STATEMENT"
            codes.append("war_signal_review")

        return current_decision, current_category, codes

    def _maybe_reload_runtime_config(self) -> None:
        cfg = self._config
        if not cfg.hot_reload_enabled or not self._config_path:
            return
        now = time.time()
        if now - self._last_reload_check < max(cfg.hot_reload_poll_seconds, 1.0):
            return
        self._last_reload_check = now
        try:
            from pathlib import Path

            import yaml

            path = Path(self._config_path)
            if not path.is_file():
                return
            raw = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
            if digest == self._config_path_hash:
                return
            payload = yaml.safe_load(raw) or {}
            section = payload.get("safety_gate", payload)
            runtime = section.get("censorship_runtime") or {}
            new_cfg = CensorRuntimeConfig(**runtime)
            # Atomic swap; in-flight request keeps local snapshot "cfg".
            self._config = new_cfg
            self._config_path_hash = digest
            self._refresh_version_hashes()
            logger.info("pre_rag_censor_config_reloaded hash=%s", digest)
        except Exception as error:  # noqa: BLE001
            logger.warning("pre_rag_censor_config_reload_failed err=%s", error)

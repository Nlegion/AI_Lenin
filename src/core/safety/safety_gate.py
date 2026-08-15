"""SafetyGate: isolated pre-LLM censorship with shadow/dual-run support."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from src.core.safety.news_guard import InputGateResult, NewsGuard
from src.core.safety.safety_gate_types import (
    GateContext,
    GateDecision,
    RuleResult,
    SafetyHint,
    ShadowCompareResult,
)
from src.core.settings.safety_gate_config import (
    SafetyGateConfig,
    default_safety_gate_config_path,
    load_safety_gate_config,
    safety_gate_version_hash,
)

logger = logging.getLogger(__name__)


def _hints_for_tier(risk_tier: str) -> list[SafetyHint]:
    if risk_tier == "yellow":
        return [
            SafetyHint.YELLOW_CONSTRAINED_ANALYSIS,
            SafetyHint.AVOID_COMBAT_ESTIMATES,
        ]
    return []


def _from_input_gate(
    result: InputGateResult, *, latency_ms: float, trace: dict[str, Any]
) -> GateDecision:
    hints = _hints_for_tier(result.risk_tier)
    return GateDecision(
        decision=result.decision,
        risk_tier=result.risk_tier,
        reason=result.reason,
        reason_codes=list(result.reason_codes),
        message=result.message,
        context_hints=hints,
        trace=trace,
        latency_ms=latency_ms,
        needs_yellow_warning=result.risk_tier == "yellow"
        and result.decision == "allow",
    )


class SafetyGate:
    """Rule-composition gate. Delegates heuristics via NewsGuard during migration."""

    def __init__(
        self,
        *,
        config: SafetyGateConfig,
        news_guard: NewsGuard,
        config_version_hash: str = "",
    ):
        self.config = config
        self.news_guard = news_guard
        self.config_version_hash = config_version_hash
        self._cache: OrderedDict[str, GateDecision] = OrderedDict()
        self._fallback_keys_used = list(config.fallback_keys_used)

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "SafetyGate":
        sg_path = default_safety_gate_config_path(base_dir)
        ng_path = base_dir / "config" / "news_guard.yaml"
        cfg = load_safety_gate_config(path=sg_path, news_guard_path=ng_path)
        news_guard = NewsGuard.from_file(path=ng_path)
        return cls(
            config=cfg,
            news_guard=news_guard,
            config_version_hash=safety_gate_version_hash(sg_path),
        )

    def evaluate(self, ctx: GateContext) -> GateDecision:
        started = time.perf_counter()
        cache_key = self._cache_key(ctx)
        if self.config.flags.cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        decision = self._evaluate_uncached(ctx=ctx, started=started)
        if self.config.flags.cache_enabled:
            self._cache[cache_key] = decision
            while len(self._cache) > max(self.config.flags.cache_max_entries, 1):
                self._cache.popitem(last=False)
        return decision

    def evaluate_with_shadow(
        self,
        ctx: GateContext,
        *,
        legacy_guard: NewsGuard | None = None,
    ) -> ShadowCompareResult:
        """Run new + old paths; enforce according to flags.enforce_mode."""
        new_decision = self.evaluate(ctx)
        old_decision: GateDecision | None = None
        guard = legacy_guard or self.news_guard
        started = time.perf_counter()
        legacy = guard.evaluate_input(ctx.title, ctx.content, source=ctx.source)
        old_decision = _from_input_gate(
            legacy,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            trace={"path": "legacy_news_guard", "rule": "evaluate_input"},
        )
        match = (
            old_decision.decision == new_decision.decision
            and old_decision.risk_tier == new_decision.risk_tier
        )
        reason_diff: list[str] = []
        if old_decision.decision != new_decision.decision:
            reason_diff.append(
                f"decision:{old_decision.decision}->{new_decision.decision}"
            )
        if set(old_decision.reason_codes) != set(new_decision.reason_codes):
            reason_diff.append(
                "codes:"
                f"{sorted(set(old_decision.reason_codes) ^ set(new_decision.reason_codes))}"
            )
        if not match:
            logger.info(
                "safety_gate_shadow_mismatch item_id=%s old=%s new=%s diff=%s",
                ctx.item_id,
                old_decision.decision,
                new_decision.decision,
                ",".join(reason_diff),
            )
        enforce_new = (
            self.config.flags.enabled
            and self.config.flags.enforce_mode == "new"
            and not self.config.flags.shadow_mode
        )
        # Shadow mode always enforces old path.
        if self.config.flags.shadow_mode or not self.config.flags.enabled:
            enforced = old_decision
        elif enforce_new:
            enforced = new_decision
        else:
            enforced = old_decision
        return ShadowCompareResult(
            enforced=enforced,
            old_decision=old_decision,
            new_decision=new_decision,
            decision_match=match,
            reason_diff=reason_diff,
            config_version_hash=self.config_version_hash,
        )

    def _evaluate_uncached(self, *, ctx: GateContext, started: float) -> GateDecision:
        # Migration path: compose via standardized rule wrappers over NewsGuard.
        rules = (
            self._rule_drone_combat,
            self._rule_military_risk,
            self._rule_fio_pii,
            self._rule_topic_route,
            self._rule_quarantine_unknown,
        )
        # Full evaluate_input already encodes ordered policy; wrap once for parity.
        legacy = self.news_guard.evaluate_input(
            ctx.title, ctx.content, source=ctx.source
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        rule_hits: list[str] = []
        for rule in rules:
            hit = rule(ctx)
            if hit.hit:
                rule_hits.append(
                    hit.reason or hit.reason_codes[0] if hit.reason_codes else "hit"
                )
        return _from_input_gate(
            legacy,
            latency_ms=latency_ms,
            trace={
                "path": "safety_gate",
                "config_version_hash": self.config_version_hash,
                "fallback_keys_used": list(self._fallback_keys_used),
                "rule_hits": rule_hits,
                "item_id": ctx.item_id,
                "pipeline_id": ctx.pipeline_id,
            },
        )

    def _rule_drone_combat(self, ctx: GateContext) -> RuleResult:
        from src.core.safety.drone_combat_guard import drone_air_raid_hit
        from src.core.safety.hotfix_flags import safety_flag_enabled

        if not safety_flag_enabled("drone_deny_enabled"):
            return RuleResult(hit=False)
        hit = drone_air_raid_hit(f"{ctx.title}\n{ctx.content}")
        if not hit.hit:
            return RuleResult(hit=False)
        return RuleResult(
            hit=True,
            decision="deny",
            risk_tier="red",
            reason="drone/air-raid hard deny matched",
            reason_codes=list(hit.codes),
            message=self.config.policy.refusal_message,
        )

    def _rule_military_risk(self, ctx: GateContext) -> RuleResult:
        from src.core.safety.combat_detect import (
            combat_cooccurrence_hit,
            military_rf_context_hit,
        )
        from src.core.safety.risk_routing import strong_military_hits

        text = f"{ctx.title}\n{ctx.content}"
        combat = combat_cooccurrence_hit(text)
        military_rf = military_rf_context_hit(text.lower())
        strong = strong_military_hits(text)
        if combat or military_rf or strong:
            return RuleResult(
                hit=True,
                decision="deny",
                risk_tier="red",
                reason="military/combat topic hard deny matched",
                reason_codes=list(combat) + list(strong),
                message=self.config.policy.refusal_message,
            )
        return RuleResult(hit=False)

    def _rule_fio_pii(self, ctx: GateContext) -> RuleResult:
        from src.core.safety.fio_guards import fio_spans, should_block_fio

        codes = should_block_fio(
            text=f"{ctx.title}\n{ctx.content}",
            matches=fio_spans(f"{ctx.title}\n{ctx.content}"),
        )
        if not codes:
            return RuleResult(hit=False)
        return RuleResult(
            hit=True,
            decision="deny",
            risk_tier="red",
            reason="private pii detected without public-interest context",
            reason_codes=list(codes),
            message=self.config.policy.refusal_message,
        )

    def _rule_topic_route(self, ctx: GateContext) -> RuleResult:
        from src.core.safety.topic_routing import route_topic

        routed = route_topic(title=ctx.title, content=ctx.content)
        if routed.route == "skip":
            return RuleResult(
                hit=True,
                decision="skip",
                risk_tier="green",
                reason="out-of-scope primary topic",
                reason_codes=list(routed.reason_codes),
                message=self.config.policy.skip_message,
            )
        if routed.route == "full":
            return RuleResult(
                hit=True,
                decision="allow",
                risk_tier="green",
                reason="topic route full path",
                reason_codes=list(routed.reason_codes),
            )
        return RuleResult(hit=False)

    def _rule_quarantine_unknown(self, ctx: GateContext) -> RuleResult:
        # Marker rule: unknown topic handling remains in NewsGuard evaluate_input.
        return RuleResult(hit=False, reason="deferred_to_news_guard_unknown")

    def _cache_key(self, ctx: GateContext) -> str:
        raw = f"{ctx.title}\n{ctx.content}\n{ctx.source or ''}\n{self.config_version_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def apply_yellow_warning(
    *, analysis: str, decision: GateDecision, warning_text: str
) -> str:
    """Inject yellow warning into analysis body (upstream of publisher)."""
    if not decision.needs_yellow_warning:
        return analysis
    warn = (warning_text or "").strip()
    if not warn or warn in analysis:
        return analysis
    body = analysis.strip()
    return f"{body}\n\n{warn}".strip()

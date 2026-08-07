"""NewsGate and NewsGuard safety layer for legal-risk mitigation.

Migration note (SafetyGate Stage 1+):
- Do not add new business rules here; land new policy in ``SafetyGate``.
- Critical bug-fixes in existing heuristics must be mirrored into SafetyGate
  with parity tests in the same change window.
- This module remains a temporary compatibility / output-guard adapter until
  Stage 3 freeze removes the dual path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, Field
import yaml

from src.core.safety.combat_detect import combat_cooccurrence_hit, military_rf_context_hit
from src.core.safety.drone_combat_guard import drone_air_raid_hit
from src.core.safety.fio_guards import fio_spans, should_block_fio
from src.core.safety.pattern_match import pattern_hits
from src.core.safety.risk_routing import (
    RiskTier,
    map_decision_to_tier,
    strong_military_hits,
    yellow_economy_eligible,
)
from src.core.safety.topic_routing import route_topic
from src.core.safety.hotfix_flags import safety_flag_enabled

Decision = Literal["allow", "deny", "quarantine", "skip"]

SKIP_MESSAGE = "Тема вне сферы марксистско-ленинского анализа новостей."


class DisclaimerConfig(BaseModel):
    enabled: bool = True
    placement: Literal["header", "footer"] = "footer"
    text: str


class CombatConfig(BaseModel):
    window_tokens: int = 10
    combat_stems: list[str] = Field(default_factory=list)
    military_co_tokens: list[str] = Field(default_factory=list)


class InputGateConfig(BaseModel):
    enabled: bool = True
    allow_topics: list[str] = Field(default_factory=list)
    hard_deny_topics: list[str] = Field(default_factory=list)
    quarantine_topics: list[str] = Field(default_factory=list)
    hard_deny_keywords: list[str] = Field(default_factory=list)
    quarantine_keywords: list[str] = Field(default_factory=list)
    military_topics: list[str] = Field(default_factory=list)
    trusted_sources: list[str] = Field(default_factory=list)
    high_risk_topics: list[str] = Field(default_factory=list)
    block_private_pii: bool = True
    public_interest_topics: list[str] = Field(default_factory=list)
    refusal_message: str = "Анализ данной темы невозможен в соответствии с политикой безопасности."
    skip_message: str = SKIP_MESSAGE
    classify_on_unknown_as: Decision = "quarantine"
    combat: CombatConfig = Field(default_factory=CombatConfig)
    economy_policy_markers: list[str] = Field(default_factory=list)
    yellow_block_patterns: list[str] = Field(default_factory=list)


class OutputGuardConfig(BaseModel):
    enabled: bool = True
    safe_mode: Literal["strict", "moderate", "off"] = "strict"
    block_patterns: list[str] = Field(default_factory=list)
    rewrite_patterns: list[str] = Field(default_factory=list)
    pii_patterns: list[str] = Field(default_factory=list)
    classifier_keywords: list[str] = Field(default_factory=list)
    classifier_threshold: int = 1
    hallucination_notice: str = "В стилизованной интерпретации"
    safe_template: str = "Данная тема не входит в сферу марксистско-ленинского анализа."


class NewsGuardConfig(BaseModel):
    policy_version: str = "1.0.0"
    input_gate: InputGateConfig
    output_guard: OutputGuardConfig
    disclaimer: DisclaimerConfig


@dataclass(frozen=True)
class InputGateResult:
    decision: Decision
    reason: str
    reason_codes: list[str]
    message: str = "Анализ данной темы невозможен в соответствии с политикой безопасности."
    risk_tier: RiskTier = "green"


@dataclass(frozen=True)
class OutputGuardResult:
    blocked: bool
    moderated_text: str
    reason_codes: list[str]


def load_news_guard_config(path: Path) -> NewsGuardConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("news_guard", payload)
    return NewsGuardConfig.model_validate(section)


def _extract_pii_hits(text: str, patterns: list[str]) -> list[str]:
    """Match PII patterns. FIO stays case-sensitive."""
    hits: list[str] = []
    for pattern in patterns:
        flags = 0
        if "@" in pattern or r"\d" in pattern or "ул" in pattern.lower():
            flags = re.IGNORECASE
        if re.search(pattern, text, flags=flags):
            hits.append(pattern)
    return hits


def _is_fio_pattern(pattern: str) -> bool:
    return "[А-ЯЁ]" in pattern or "[А-Яа-я]" in pattern


class NewsGuard:
    def __init__(self, config: NewsGuardConfig):
        self.config = config

    @classmethod
    def from_file(cls, path: Path) -> "NewsGuard":
        return cls(config=load_news_guard_config(path=path))

    def evaluate_input(self, title: str, content: str, source: str | None = None) -> InputGateResult:
        if not self.config.input_gate.enabled:
            return InputGateResult(
                decision="allow",
                reason="input gate disabled",
                reason_codes=[],
                message="",
                risk_tier="green",
            )

        original = f"{title}\n{content}"
        text = original.lower()
        gate = self.config.input_gate
        refuse = gate.refusal_message

        combat_hits = combat_cooccurrence_hit(
            original,
            combat_stems=gate.combat.combat_stems or None,
            co_tokens=gate.combat.military_co_tokens or None,
            window=gate.combat.window_tokens,
        )
        drone_hit = (
            drone_air_raid_hit(original)
            if safety_flag_enabled("drone_deny_enabled")
            else None
        )
        if drone_hit is not None and drone_hit.hit:
            return InputGateResult(
                decision="deny",
                reason="drone/air-raid hard deny matched",
                reason_codes=list(drone_hit.codes),
                message=refuse,
                risk_tier="red",
            )
        military_rf = military_rf_context_hit(text)
        strong_hits = strong_military_hits(text)
        military_hits = pattern_hits(text=text, patterns=self._military_topics())
        if military_rf:
            military_hits = [*military_hits, "context:military_rf_forces"]
        hard_red = bool(combat_hits or military_rf or strong_hits)
        if hard_red:
            return InputGateResult(
                decision="deny",
                reason="military/combat topic hard deny matched",
                reason_codes=combat_hits + strong_hits + military_hits,
                message=refuse,
                risk_tier="red",
            )
        if military_hits:
            eligible, econ = yellow_economy_eligible(
                text=original,
                combat_hits=combat_hits,
                military_rf=military_rf,
                strong_military=strong_hits,
                other_red=[],
                economy_markers=gate.economy_policy_markers or None,
            )
            if eligible:
                return InputGateResult(
                    decision="allow",
                    reason="economy yellow carve-out (weak military lexical)",
                    reason_codes=[*econ, *military_hits, "risk_tier:yellow"],
                    message="",
                    risk_tier="yellow",
                )
            return InputGateResult(
                decision="deny",
                reason="military/combat topic hard deny matched",
                reason_codes=military_hits,
                message=refuse,
                risk_tier="red",
            )

        if gate.trusted_sources and source:
            normalized_source = source.strip().lower()
            trusted = {item.strip().lower() for item in gate.trusted_sources}
            high_risk = pattern_hits(text=text, patterns=gate.high_risk_topics)
            if normalized_source not in trusted and high_risk:
                return InputGateResult(
                    decision="deny",
                    reason="source not in trusted list for high-risk topic",
                    reason_codes=high_risk + [f"source:{source}"],
                    message="Источник новости не входит в перечень доверенных изданий.",
                    risk_tier="red",
                )

        from src.core.safety.fio_guards import has_public_interest_context, is_private_victim_context

        fio_codes = should_block_fio(text=original, matches=fio_spans(original))
        public_interest = pattern_hits(text=text, patterns=gate.public_interest_topics)
        if safety_flag_enabled("fio_carveout_enabled") and has_public_interest_context(original):
            public_interest = [*public_interest, "public_interest:stem_or_role"]
        other_pii = [
            p
            for p in _extract_pii_hits(text=original, patterns=self._pii_patterns())
            if not _is_fio_pattern(p)
        ]
        if is_private_victim_context(original) and not public_interest:
            return InputGateResult(
                decision="skip",
                reason="private victim/relative context soft-skip",
                reason_codes=["private_victim_context"],
                message=self._skip_message(primary="default"),
                risk_tier="yellow",
            )
        if gate.block_private_pii and not public_interest and (fio_codes or other_pii):
            return InputGateResult(
                decision="deny",
                reason="private pii detected without public-interest context",
                reason_codes=fio_codes + other_pii,
                message=refuse,
                risk_tier="red",
            )

        hard_kw = pattern_hits(text=text, patterns=gate.hard_deny_keywords)
        if hard_kw:
            return InputGateResult(
                decision="deny",
                reason="hard deny keyword matched",
                reason_codes=hard_kw,
                message=refuse,
                risk_tier="red",
            )

        sport_negatives = self._sport_intra_negatives()
        routed = route_topic(
            title=title,
            content=content,
            sport_intra_negatives=sport_negatives,
        )
        if routed.route == "skip":
            return InputGateResult(
                decision="skip",
                reason="out-of-scope primary topic",
                reason_codes=routed.reason_codes,
                message=self._skip_message(primary=routed.primary),
                risk_tier="green",
            )
        if routed.route == "full":
            return InputGateResult(
                decision="allow",
                reason="topic route full path",
                reason_codes=routed.reason_codes,
                message="",
                risk_tier="green",
            )

        hard_topics = pattern_hits(text=text, patterns=gate.hard_deny_topics)
        if hard_topics:
            return InputGateResult(
                decision="skip",
                reason="content-type out of scope",
                reason_codes=hard_topics,
                message=self._skip_message(primary="default"),
                risk_tier="green",
            )

        quarantine = pattern_hits(text=text, patterns=gate.quarantine_topics + gate.quarantine_keywords)
        if quarantine:
            eligible, econ = yellow_economy_eligible(
                text=original,
                combat_hits=[],
                military_rf=False,
                strong_military=[],
                other_red=[],
                economy_markers=gate.economy_policy_markers or None,
            )
            if eligible:
                return InputGateResult(
                    decision="allow",
                    reason="economy yellow carve-out from quarantine",
                    reason_codes=[*econ, *quarantine, "risk_tier:yellow"],
                    message="",
                    risk_tier="yellow",
                )
            return InputGateResult(
                decision="quarantine",
                reason="quarantine topic/keyword matched",
                reason_codes=quarantine,
                message=refuse,
                risk_tier="yellow",
            )

        allow_hits = pattern_hits(text=text, patterns=gate.allow_topics)
        if allow_hits:
            return InputGateResult(
                decision="allow",
                reason="allow topic matched",
                reason_codes=allow_hits,
                message="",
                risk_tier="green",
            )

        eligible, econ = yellow_economy_eligible(
            text=original,
            combat_hits=[],
            military_rf=False,
            strong_military=[],
            other_red=[],
            economy_markers=gate.economy_policy_markers or None,
        )
        if eligible:
            return InputGateResult(
                decision="allow",
                reason="economy yellow carve-out for unknown topic",
                reason_codes=[*econ, "risk_tier:yellow"],
                message="",
                risk_tier="yellow",
            )

        unknown = InputGateResult(
            decision=gate.classify_on_unknown_as,
            reason="no explicit allow topic matched",
            reason_codes=["unknown_topic"],
            message=refuse,
            risk_tier=map_decision_to_tier(gate.classify_on_unknown_as),
        )
        return unknown

    def guard_output(
        self,
        analysis: str,
        source_text: str | None = None,
        warn_only: bool = False,
        *,
        risk_tier: RiskTier | None = None,
        extra_block_patterns: list[str] | None = None,
    ) -> OutputGuardResult:
        if not self.config.output_guard.enabled:
            return OutputGuardResult(blocked=False, moderated_text=self._apply_disclaimer(analysis), reason_codes=[])

        text = analysis
        reason_codes: list[str] = []
        classifier_hits = self._classify_extremism(text=text)
        if classifier_hits:
            reason_codes.extend([f"classifier:{item}" for item in classifier_hits])
        block_patterns = list(self.config.output_guard.block_patterns)
        if risk_tier == "yellow":
            block_patterns.extend(self.config.input_gate.yellow_block_patterns)
            if extra_block_patterns:
                block_patterns.extend(extra_block_patterns)
        for pattern in block_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                reason_codes.append(f"block:{pattern}")

        if reason_codes and self.config.output_guard.safe_mode == "strict" and not warn_only:
            safe_text = self._apply_disclaimer(self.config.output_guard.safe_template)
            return OutputGuardResult(blocked=True, moderated_text=safe_text, reason_codes=reason_codes)

        for pattern in self.config.output_guard.rewrite_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                reason_codes.append(f"rewrite:{pattern}")
                if self.config.output_guard.safe_mode == "moderate":
                    text = re.sub(pattern, "[отредактировано]", text, flags=re.IGNORECASE)

        # Redact FIO without IGNORECASE (case-sensitive sub); other PII may use IGNORECASE.
        pii_hits = _extract_pii_hits(text=text, patterns=self._pii_patterns(output=True))
        if pii_hits and source_text:
            source_pii = set(_extract_pii_hits(text=source_text, patterns=self._pii_patterns(output=True)))
            for pattern in pii_hits:
                if pattern in source_pii:
                    continue
                flags = 0 if _is_fio_pattern(pattern) else re.IGNORECASE
                text = re.sub(pattern, "«[место]»", text, flags=flags)
                reason_codes.append(f"pii_redact:{pattern}")

        if "[обезличено]" in text or "«[место]»" in text:
            reason_codes.append("redact_artifact_present")

        return OutputGuardResult(
            blocked=False,
            moderated_text=self._apply_disclaimer(text),
            reason_codes=reason_codes,
        )

    def mark_unverified_facts(self, analysis: str, retrieval_context: str) -> tuple[str, list[str]]:
        reason_codes: list[str] = []
        context_lower = retrieval_context.lower()
        lines = [line.strip() for line in analysis.split(".") if line.strip()]
        updated: list[str] = []
        for line in lines:
            lowered = line.lower()
            looks_factual = '"' in line or any(char.isdigit() for char in line) or "как я писал" in lowered
            if looks_factual and lowered not in context_lower:
                updated.append(f"{self.config.output_guard.hallucination_notice}: {line}")
                reason_codes.append("hallucination_marked")
            else:
                updated.append(line)
        merged = ". ".join(updated).strip()
        if merged and not merged.endswith("."):
            merged += "."
        return merged or analysis, reason_codes

    def _military_topics(self) -> list[str]:
        # Bare «сво» handled via phrase/token rules in pattern_hits; keep phrase forms here.
        defaults = [
            "вс рф",
            "вооруженные силы",
            "вооружённые силы",
            "росгварди",
            "специальной военной операции",
            "мобилизац",
            "министерство обороны",
            "боевые действия",
            "армия россии",
            "сво",
        ]
        return list(dict.fromkeys([*defaults, *self.config.input_gate.military_topics]))

    def _quality_config(self):
        from src.core.settings.quality_postcheck_config import (
            default_quality_postcheck_path,
            load_quality_postcheck_config,
        )

        root = Path(__file__).resolve().parents[3]
        path = default_quality_postcheck_path(root)
        if not path.is_file():
            return None
        return load_quality_postcheck_config(path=path)

    def _sport_intra_negatives(self) -> list[str]:
        cfg = self._quality_config()
        return list(cfg.sport_intra_negatives) if cfg is not None else []

    def _skip_message(self, *, primary: str) -> str:
        from src.core.safety.skip_templates import skip_message_for_primary

        cfg = self._quality_config()
        if cfg is None:
            return self.config.input_gate.skip_message
        return skip_message_for_primary(primary=primary, config=cfg)

    def _pii_patterns(self, output: bool = False) -> list[str]:
        defaults = [
            r"\b\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b",
            r"\b\d{10,12}\b",
            r"\b[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+\b",
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            r"\bул\.?\s+[А-Яа-яA-Za-z0-9\-\s]+,\s*д\.?\s*\d+\b",
        ]
        configured = self.config.output_guard.pii_patterns if output else []
        return list(dict.fromkeys([*defaults, *configured]))

    def _classify_extremism(self, text: str) -> list[str]:
        defaults = [
            "к оружию",
            "свержение власти",
            "насильственное изменение конституционного строя",
            "террор",
            "экстремист",
            "разжиг",
        ]
        keywords = list(dict.fromkeys([*defaults, *self.config.output_guard.classifier_keywords]))
        lowered = text.lower()
        hits = [item for item in keywords if item in lowered]
        if len(hits) >= max(self.config.output_guard.classifier_threshold, 1):
            return hits
        return []

    def _apply_disclaimer(self, text: str) -> str:
        from src.core.generation.output_artifacts import LONG_DISCLAIMER_RE
        from src.core.safety.hotfix_flags import generation_flag_enabled

        working = text
        if generation_flag_enabled("disclaimer_footer_enabled"):
            working = LONG_DISCLAIMER_RE.sub("", working).strip()
            # Prefer short footer from quality config when present.
            cfg = self._quality_config()
            disclaimer = (
                (cfg.short_disclaimer if cfg is not None else "")
                or self.config.disclaimer.text
            ).strip()
            if not self.config.disclaimer.enabled and not disclaimer:
                return working
            body = working.strip()
            if disclaimer and disclaimer in body:
                return body
            return f"{body}\n\n{disclaimer}".strip() if disclaimer else body

        if not self.config.disclaimer.enabled:
            return text
        disclaimer = self.config.disclaimer.text.strip()
        if self.config.disclaimer.placement == "header":
            return f"{disclaimer}\n\n{text}".strip()
        return f"{text.strip()}\n\n{disclaimer}".strip()

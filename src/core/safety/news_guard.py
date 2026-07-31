"""NewsGate and NewsGuard safety layer for legal-risk mitigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, Field
import yaml

from src.core.safety.combat_detect import combat_cooccurrence_hit, military_rf_context_hit
from src.core.safety.fio_guards import fio_spans, should_block_fio
from src.core.safety.pattern_match import pattern_hits
from src.core.safety.topic_routing import route_topic

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
            return InputGateResult(decision="allow", reason="input gate disabled", reason_codes=[], message="")

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
        military_hits = pattern_hits(text=text, patterns=self._military_topics())
        if military_rf_context_hit(text):
            military_hits = [*military_hits, "context:military_rf_forces"]
        if combat_hits or military_hits:
            return InputGateResult(
                decision="deny",
                reason="military/combat topic hard deny matched",
                reason_codes=combat_hits + military_hits,
                message=refuse,
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
                )

        fio_codes = should_block_fio(text=original, matches=fio_spans(original))
        public_interest = pattern_hits(text=text, patterns=gate.public_interest_topics)
        other_pii = [
            p
            for p in _extract_pii_hits(text=original, patterns=self._pii_patterns())
            if not _is_fio_pattern(p)
        ]
        if gate.block_private_pii and not public_interest and (fio_codes or other_pii):
            return InputGateResult(
                decision="deny",
                reason="private pii detected without public-interest context",
                reason_codes=fio_codes + other_pii,
                message=refuse,
            )

        hard_kw = pattern_hits(text=text, patterns=gate.hard_deny_keywords)
        if hard_kw:
            return InputGateResult(
                decision="deny",
                reason="hard deny keyword matched",
                reason_codes=hard_kw,
                message=refuse,
            )

        routed = route_topic(title=title, content=content)
        if routed.route == "skip":
            return InputGateResult(
                decision="skip",
                reason="out-of-scope primary topic",
                reason_codes=routed.reason_codes,
                message=gate.skip_message,
            )
        if routed.route == "full":
            return InputGateResult(
                decision="allow",
                reason="topic route full path",
                reason_codes=routed.reason_codes,
                message="",
            )

        # Remaining hard_deny content-types (non-sport handled by router when possible)
        hard_topics = pattern_hits(text=text, patterns=gate.hard_deny_topics)
        if hard_topics:
            return InputGateResult(
                decision="skip",
                reason="content-type out of scope",
                reason_codes=hard_topics,
                message=gate.skip_message,
            )

        quarantine = pattern_hits(text=text, patterns=gate.quarantine_topics + gate.quarantine_keywords)
        if quarantine:
            return InputGateResult(
                decision="quarantine",
                reason="quarantine topic/keyword matched",
                reason_codes=quarantine,
                message=refuse,
            )

        allow_hits = pattern_hits(text=text, patterns=gate.allow_topics)
        if allow_hits:
            return InputGateResult(decision="allow", reason="allow topic matched", reason_codes=allow_hits, message="")

        return InputGateResult(
            decision=gate.classify_on_unknown_as,
            reason="no explicit allow topic matched",
            reason_codes=["unknown_topic"],
            message=refuse,
        )

    def guard_output(self, analysis: str, source_text: str | None = None, warn_only: bool = False) -> OutputGuardResult:
        if not self.config.output_guard.enabled:
            return OutputGuardResult(blocked=False, moderated_text=self._apply_disclaimer(analysis), reason_codes=[])

        text = analysis
        reason_codes: list[str] = []
        classifier_hits = self._classify_extremism(text=text)
        if classifier_hits:
            reason_codes.extend([f"classifier:{item}" for item in classifier_hits])
        for pattern in self.config.output_guard.block_patterns:
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
                text = re.sub(pattern, "[обезличено]", text, flags=flags)
                reason_codes.append(f"pii_redact:{pattern}")

        if "[обезличено]" in text:
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
        if not self.config.disclaimer.enabled:
            return text
        disclaimer = self.config.disclaimer.text.strip()
        if self.config.disclaimer.placement == "header":
            return f"{disclaimer}\n\n{text}".strip()
        return f"{text.strip()}\n\n{disclaimer}".strip()

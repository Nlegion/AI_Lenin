"""NewsGate and NewsGuard safety layer for legal-risk mitigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, Field
import yaml


Decision = Literal["allow", "deny", "quarantine"]


class DisclaimerConfig(BaseModel):
    enabled: bool = True
    placement: Literal["header", "footer"] = "footer"
    text: str


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
    classify_on_unknown_as: Decision = "quarantine"


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


def _contains_any(text: str, patterns: list[str]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for pattern in patterns:
        if pattern.lower() in lowered:
            hits.append(pattern)
    return hits


def _extract_pii_hits(text: str, patterns: list[str]) -> list[str]:
    """Match PII patterns. Do not use IGNORECASE for Cyrillic FIO (would match any 3 words)."""
    hits: list[str] = []
    for pattern in patterns:
        flags = 0
        if "@" in pattern or r"\d" in pattern or "ул" in pattern.lower():
            flags = re.IGNORECASE
        if re.search(pattern, text, flags=flags):
            hits.append(pattern)
    return hits


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
        military_hits = _contains_any(text=text, patterns=self._military_topics())
        if self._military_context_hit(text=text):
            military_hits.append("context:military_rf_forces")
        if military_hits:
            return InputGateResult(
                decision="deny",
                reason="military topic hard deny matched",
                reason_codes=military_hits,
                message=self.config.input_gate.refusal_message,
            )

        if self.config.input_gate.trusted_sources and source:
            normalized_source = source.strip().lower()
            normalized_trusted = {item.strip().lower() for item in self.config.input_gate.trusted_sources}
            high_risk_hits = _contains_any(text=text, patterns=self.config.input_gate.high_risk_topics)
            if normalized_source not in normalized_trusted and high_risk_hits:
                return InputGateResult(
                    decision="deny",
                    reason="source not in trusted list for high-risk topic",
                    reason_codes=high_risk_hits + [f"source:{source}"],
                    message="Источник новости не входит в перечень доверенных изданий.",
                )

        # FIO pattern is case-sensitive; must run on original casing (not lowercased text).
        pii_hits = _extract_pii_hits(text=original, patterns=self._pii_patterns())
        public_interest_hits = _contains_any(text=text, patterns=self.config.input_gate.public_interest_topics)
        if pii_hits and self.config.input_gate.block_private_pii and not public_interest_hits:
            return InputGateResult(
                decision="deny",
                reason="private pii detected without public-interest context",
                reason_codes=pii_hits,
                message=self.config.input_gate.refusal_message,
            )

        hard_deny_topic_hits = _contains_any(text=text, patterns=self.config.input_gate.hard_deny_topics)
        hard_deny_keyword_hits = _contains_any(text=text, patterns=self.config.input_gate.hard_deny_keywords)
        if hard_deny_topic_hits or hard_deny_keyword_hits:
            return InputGateResult(
                decision="deny",
                reason="hard deny topic/keyword matched",
                reason_codes=hard_deny_topic_hits + hard_deny_keyword_hits,
                message=self.config.input_gate.refusal_message,
            )

        quarantine_topic_hits = _contains_any(text=text, patterns=self.config.input_gate.quarantine_topics)
        quarantine_keyword_hits = _contains_any(text=text, patterns=self.config.input_gate.quarantine_keywords)
        if quarantine_topic_hits or quarantine_keyword_hits:
            return InputGateResult(
                decision="quarantine",
                reason="quarantine topic/keyword matched",
                reason_codes=quarantine_topic_hits + quarantine_keyword_hits,
                message=self.config.input_gate.refusal_message,
            )

        allow_hits = _contains_any(text=text, patterns=self.config.input_gate.allow_topics)
        if allow_hits:
            return InputGateResult(decision="allow", reason="allow topic matched", reason_codes=allow_hits, message="")

        return InputGateResult(
            decision=self.config.input_gate.classify_on_unknown_as,
            reason="no explicit allow topic matched",
            reason_codes=["unknown_topic"],
            message=self.config.input_gate.refusal_message,
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

        pii_hits = _extract_pii_hits(text=text, patterns=self._pii_patterns(output=True))
        if pii_hits and source_text:
            source_pii_hits = set(_extract_pii_hits(text=source_text, patterns=self._pii_patterns(output=True)))
            for pattern in pii_hits:
                if pattern in source_pii_hits:
                    continue
                text = re.sub(pattern, "[обезличено]", text, flags=re.IGNORECASE)
                reason_codes.append(f"pii_redact:{pattern}")

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
        defaults = [
            "вс рф",
            "вооруженные силы",
            "вооружённые силы",
            "росгварди",
            "сво",
            "специальной военной операции",
            "мобилизац",
            "министерство обороны",
            "боевые действия",
            "армия россии",
        ]
        return list(dict.fromkeys([*defaults, *self.config.input_gate.military_topics]))

    @staticmethod
    def _military_context_hit(text: str) -> bool:
        patterns = [
            r"(военн\w+|арм\w+|силов\w+).{0,40}(рф|росси\w+)",
            r"(рф|росси\w+).{0,40}(военн\w+|арм\w+|силов\w+)",
        ]
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

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

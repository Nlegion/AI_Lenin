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
    classify_on_unknown_as: Decision = "quarantine"


class OutputGuardConfig(BaseModel):
    enabled: bool = True
    safe_mode: Literal["strict", "moderate", "off"] = "strict"
    block_patterns: list[str] = Field(default_factory=list)
    rewrite_patterns: list[str] = Field(default_factory=list)
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


class NewsGuard:
    def __init__(self, config: NewsGuardConfig):
        self.config = config

    @classmethod
    def from_file(cls, path: Path) -> "NewsGuard":
        return cls(config=load_news_guard_config(path=path))

    def evaluate_input(self, title: str, content: str) -> InputGateResult:
        if not self.config.input_gate.enabled:
            return InputGateResult(decision="allow", reason="input gate disabled", reason_codes=[])

        text = f"{title}\n{content}".lower()
        hard_deny_topic_hits = _contains_any(text=text, patterns=self.config.input_gate.hard_deny_topics)
        hard_deny_keyword_hits = _contains_any(text=text, patterns=self.config.input_gate.hard_deny_keywords)
        if hard_deny_topic_hits or hard_deny_keyword_hits:
            return InputGateResult(
                decision="deny",
                reason="hard deny topic/keyword matched",
                reason_codes=hard_deny_topic_hits + hard_deny_keyword_hits,
            )

        quarantine_topic_hits = _contains_any(text=text, patterns=self.config.input_gate.quarantine_topics)
        quarantine_keyword_hits = _contains_any(text=text, patterns=self.config.input_gate.quarantine_keywords)
        if quarantine_topic_hits or quarantine_keyword_hits:
            return InputGateResult(
                decision="quarantine",
                reason="quarantine topic/keyword matched",
                reason_codes=quarantine_topic_hits + quarantine_keyword_hits,
            )

        allow_hits = _contains_any(text=text, patterns=self.config.input_gate.allow_topics)
        if allow_hits:
            return InputGateResult(decision="allow", reason="allow topic matched", reason_codes=allow_hits)

        return InputGateResult(
            decision=self.config.input_gate.classify_on_unknown_as,
            reason="no explicit allow topic matched",
            reason_codes=["unknown_topic"],
        )

    def guard_output(self, analysis: str) -> OutputGuardResult:
        if not self.config.output_guard.enabled:
            return OutputGuardResult(blocked=False, moderated_text=self._apply_disclaimer(analysis), reason_codes=[])

        text = analysis
        reason_codes: list[str] = []
        for pattern in self.config.output_guard.block_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                reason_codes.append(f"block:{pattern}")

        if reason_codes and self.config.output_guard.safe_mode == "strict":
            safe_text = self._apply_disclaimer(self.config.output_guard.safe_template)
            return OutputGuardResult(blocked=True, moderated_text=safe_text, reason_codes=reason_codes)

        for pattern in self.config.output_guard.rewrite_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                reason_codes.append(f"rewrite:{pattern}")
                if self.config.output_guard.safe_mode == "moderate":
                    text = re.sub(pattern, "[отредактировано]", text, flags=re.IGNORECASE)

        return OutputGuardResult(
            blocked=False,
            moderated_text=self._apply_disclaimer(text),
            reason_codes=reason_codes,
        )

    def _apply_disclaimer(self, text: str) -> str:
        if not self.config.disclaimer.enabled:
            return text
        disclaimer = self.config.disclaimer.text.strip()
        if self.config.disclaimer.placement == "header":
            return f"{disclaimer}\n\n{text}".strip()
        return f"{text.strip()}\n\n{disclaimer}".strip()

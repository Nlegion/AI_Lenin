"""One-pass generation artifact detect/strip (Trial50 H1)."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.core.safety.hotfix_flags import generation_flag_enabled
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

logger = logging.getLogger(__name__)

_EMPTY_PRINCIPLE = re.compile(r"принцип\s*:\s*\.", re.IGNORECASE)
_STAR_LENIN = re.compile(r"\*+\s*Ленин[^\n]{0,40}", re.IGNORECASE)
_STAR_DOT = re.compile(r"\*\s*:\s*\.")
_YEAR_ATTR = re.compile(
    r"(Ленин\s*,\s*)(19\d{2}|20\d{2})(?!\s*,?\s*т\.?\s*\d)",
    re.IGNORECASE,
)
_VOLUME_OK = re.compile(r"(том|т\.)\s*\d+", re.IGNORECASE)
_WORK_TITLE = re.compile(r"Ленин\s*,\s*[«\"]([^»\"]+)[»\"]", re.IGNORECASE)
_SCAFFOLD = re.compile(
    r"(?m)^\s*(Факт|Суть тезиса|Анализ|Вывод|Ожидаемый ответ)\s*:\s*",
    re.IGNORECASE,
)
_STYLE_LEAD = re.compile(
    r"^\s*В стилизованной интерпретаци[^\n.:]*[:.]?\s*",
    re.IGNORECASE,
)
_REDACT = re.compile(
    r"\[(обезличено|source:?[^\]]*|cite[^\]]*|redacted|removed)\]",
    re.IGNORECASE,
)
_MOJIBAKE_SG = "СЃ"
_MOJIBAKE_RYO = re.compile(r"(?<![А-Яа-яЁёA-Za-z])Рё(?![А-Яа-яЁёA-Za-z])")
_LATIN_ISLAND = re.compile(r"[а-яёА-ЯЁ]{2,}[a-zA-Z]{2,}|[a-zA-Z]{2,}[а-яёА-ЯЁ]{2,}")
_BROKEN = (
    re.compile(r"\bчто\s*,", re.IGNORECASE),
    re.compile(r"утверждение\s+о\s+и\b", re.IGNORECASE),
    re.compile(r"\bи\s+и\b", re.IGNORECASE),
)
LONG_DISCLAIMER_RE = re.compile(
    r"Ответ сгенерирован искусственным интеллектом.{0,200}призывом к действию\.?",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ArtifactPassResult:
    text: str
    codes: list[str] = field(default_factory=list)
    used_fallback: bool = False
    deny: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_encoding_artifacts(text: str) -> list[str]:
    codes: list[str] = []
    if _MOJIBAKE_SG in text:
        codes.append("artifact:mojibake_sg")
    if _MOJIBAKE_RYO.search(text):
        codes.append("artifact:mojibake_ryo")
    if _LATIN_ISLAND.search(text):
        codes.append("artifact:latin_island")
    if "\ufffd" in text:
        codes.append("artifact:replacement_char")
    return codes


def scrub_input_text(text: str) -> tuple[str, list[str]]:
    """Detect-only normalize for news/RAG; no blind СЃ→США / Рё→его."""
    if not generation_flag_enabled("encoding_scrubber_enabled"):
        return text, []
    codes = detect_encoding_artifacts(text)
    if codes:
        logger.info("artifact_detected_input codes=%s", ",".join(codes))
    return text, codes


def _pick_fallback(config: QualityPostcheckConfig, item_id: str | None) -> str:
    templates = list(getattr(config, "fallback_templates", None) or [])
    if not templates:
        templates = [config.static_safe_template, config.static_insufficient_template]
    key = (item_id or "0").encode("utf-8")
    idx = int(hashlib.md5(key).hexdigest(), 16) % len(templates)
    return templates[idx]


def _strip_citation_debris(text: str, *, grounded_titles: set[str]) -> tuple[str, list[str]]:
    codes: list[str] = []
    working = text
    if _EMPTY_PRINCIPLE.search(working):
        working = _EMPTY_PRINCIPLE.sub("", working)
        codes.append("strip:empty_principle")
    if _STAR_LENIN.search(working) and not _VOLUME_OK.search(working):
        working = _STAR_LENIN.sub("", working)
        codes.append("strip:star_lenin")
    if _STAR_DOT.search(working):
        working = _STAR_DOT.sub("", working)
        codes.append("strip:star_dot")

    def _year_sub(match: re.Match[str]) -> str:
        # Keep if nearby work title is grounded.
        start = max(0, match.start() - 80)
        window = working[start : match.end() + 80]
        title_match = _WORK_TITLE.search(window)
        if title_match and title_match.group(1).strip().lower() in grounded_titles:
            return match.group(0)
        codes.append("strip:year_only_cite")
        logger.info("year_only_cite_stripped")
        return match.group(1).rstrip(", ")

    if re.search(r"ленин", working, flags=re.IGNORECASE):
        working = _YEAR_ATTR.sub(_year_sub, working)
    return working, codes


def apply_artifact_pass(
    *,
    text: str,
    config: QualityPostcheckConfig,
    item_id: str | None = None,
    grounded_work_titles: set[str] | None = None,
    combat_sensitive: bool = False,
) -> ArtifactPassResult:
    """One normalize/detect → one strip → one fallback/deny decision."""
    codes: list[str] = []
    working = text
    meta: dict[str, Any] = {}

    soft = str(getattr(config, "artifact_enforce_mode", "soft") or "soft") == "soft"
    hard_fallback = bool(getattr(config, "hard_fallback_on_broken_output", False))
    enc = detect_encoding_artifacts(working)
    if enc and generation_flag_enabled("encoding_scrubber_enabled"):
        codes.extend(enc)
        meta["artifact_detected"] = True
        # Soft quality path: keep text, do not blind-repair or full-replace.
        if not soft and hard_fallback:
            return ArtifactPassResult(
                text=_pick_fallback(config, item_id),
                codes=[*codes, "fallback:encoding_artifact"],
                used_fallback=True,
                metadata=meta,
            )
        codes.append("detect:encoding_artifact")

    if generation_flag_enabled("loop_strip_enabled"):
        working, cite_codes = _strip_citation_debris(
            working,
            grounded_titles={t.lower() for t in (grounded_work_titles or set())},
        )
        codes.extend(cite_codes)
        if _SCAFFOLD.search(working):
            working = _SCAFFOLD.sub("", working)
            codes.append("strip:scaffold")
        if _STYLE_LEAD.search(working):
            working = _STYLE_LEAD.sub("", working)
            codes.append("strip:style_lead")
        if _REDACT.search(working):
            working = _REDACT.sub("«[место]»", working)
            codes.append("strip:redact_placeholder")
        if LONG_DISCLAIMER_RE.search(working):
            working = LONG_DISCLAIMER_RE.sub("", working)
            codes.append("strip:long_disclaimer_header")

    working = re.sub(r"\n{3,}", "\n\n", working).strip()
    broken = any(p.search(working) for p in _BROKEN)
    min_chars = int(getattr(config, "min_meaningful_chars", 40) or 40)
    too_short = len(re.sub(r"\s+", "", working)) < min_chars
    if broken:
        codes.append("broken_syntax")
    if too_short:
        codes.append("too_short_after_strip")

    # Policy deny for combat-sensitive broken output is handled by SafetyGate /
    # NewsGuard; quality pass only falls back when hard mode is explicit.
    if (broken or too_short) and hard_fallback and not soft:
        return ArtifactPassResult(
            text=_pick_fallback(config, item_id),
            codes=[*codes, "fallback:broken_or_short"],
            used_fallback=True,
            metadata=meta,
        )
    if combat_sensitive and broken and hard_fallback and not soft:
        return ArtifactPassResult(
            text="Анализ данной темы невозможен в соответствии с политикой безопасности.",
            codes=[*codes, "deny:combat_broken_output"],
            used_fallback=True,
            deny=True,
            metadata=meta,
        )

    return ArtifactPassResult(text=working, codes=codes, metadata=meta)

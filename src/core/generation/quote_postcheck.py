"""Post-generate quote grounding, attribution checks, path scrub, strip fallbacks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.core.generation.quote_allowlist import QuoteCandidate
from src.core.generation.text_normalize import normalize_for_grounding
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

logger = logging.getLogger(__name__)

_ANSWER_QUOTE = re.compile(
    r"«([^»]{3,400})»|"
    r"\"([^\"]{3,400})\"|"
    r"„([^“]{3,400})“|"
    r"“([^”]{3,400})”"
)
_ATTR_VOLUME = re.compile(r"(?i)\bтом\s*(\d+)\b")
_ATTR_PAGE = re.compile(r"(?i)\bстр\.?\s*(\d+)\b")
_ATTR_WORK = re.compile(r"(?i)(?:\*|«|\")\s*о\s+[^»\"*]{3,80}(?:\*|»|\")")
_PATH_LEAK = re.compile(
    r"\[source:\s*[^\]]+\]|"
    r"\b(?:file://|https?://)?[^\s\]\)]*(?:/pss/|\\pss\\|\.\./)[^\s\]\)]*",
    flags=re.IGNORECASE,
)
_ATTR_DEBRIS = re.compile(
    r"(?i)\b(?:как\s+(?:отмечал|писал|говорил)|по\s+словам|в\s+работе)\b[^.]{0,40}",
)
_LENIN_WROTE = re.compile(r"(?i)\bленин\s+(?:писал|сказал|отмечал)\b")


@dataclass
class QuotePostcheckResult:
    text: str
    codes: list[str] = field(default_factory=list)
    quote_allowlist_present: bool = False
    quote_removed: bool = False
    used_static_template: bool = False
    critical_attribution_hallucination: bool = False
    path_leak_scrubbed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _answer_spans(text: str) -> list[str]:
    spans: list[str] = []
    for match in _ANSWER_QUOTE.finditer(text or ""):
        span = next((g for g in match.groups() if g), None)
        if span:
            spans.append(span.strip())
    return spans


def _grounded_in_candidates(span: str, candidates: list[QuoteCandidate]) -> bool:
    norm = normalize_for_grounding(span)
    if not norm:
        return False
    for cand in candidates:
        chunk_norm = normalize_for_grounding(cand.text)
        if chunk_norm and norm in chunk_norm:
            return True
    return False


def _meta_volumes_pages(candidates: list[QuoteCandidate]) -> tuple[set[str], set[str], set[str]]:
    volumes: set[str] = set()
    pages: set[str] = set()
    works: set[str] = set()
    for cand in candidates:
        meta = cand.meta
        for key in ("volume", "том"):
            if meta.get(key) is not None:
                volumes.add(str(meta[key]).strip())
        for key in ("page", "стр", "page_start"):
            if meta.get(key) is not None:
                pages.add(str(meta[key]).strip())
        for key in ("work", "title"):
            if meta.get(key):
                works.add(normalize_for_grounding(str(meta[key])))
    return volumes, pages, works


def check_critical_attribution(text: str, candidates: list[QuoteCandidate]) -> list[str]:
    """Return codes if том/стр/work/path in answer disagree with chunk meta."""
    codes: list[str] = []
    if _PATH_LEAK.search(text or ""):
        codes.append("critical_attr:path_leak")
    volumes, pages, works = _meta_volumes_pages(candidates)
    for match in _ATTR_VOLUME.finditer(text or ""):
        vol = match.group(1)
        if volumes and vol not in volumes:
            codes.append(f"critical_attr:volume_mismatch:{vol}")
        elif not volumes:
            codes.append(f"critical_attr:volume_invented:{vol}")
    for match in _ATTR_PAGE.finditer(text or ""):
        page = match.group(1)
        if pages and page not in pages:
            codes.append(f"critical_attr:page_mismatch:{page}")
        elif not pages:
            codes.append(f"critical_attr:page_invented:{page}")
    # Invented work-like titles with year pattern often hallucinated in QA dumps.
    if re.search(r"(?i)\*\s*о\s+спорте\s*\*", text or ""):
        work_norm = normalize_for_grounding("о спорте")
        if work_norm not in works:
            codes.append("critical_attr:work_invented")
    return codes


def scrub_path_leaks(text: str) -> tuple[str, bool]:
    if not text:
        return text, False
    cleaned, n = _PATH_LEAK.subn("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, n > 0


def strip_ungrounded_quotes(
    text: str,
    *,
    candidates: list[QuoteCandidate],
) -> tuple[str, bool]:
    removed = False

    def _repl(match: re.Match[str]) -> str:
        nonlocal removed
        span = next((g for g in match.groups() if g), "")
        if span and _grounded_in_candidates(span, candidates):
            return match.group(0)
        removed = True
        return ""

    cleaned = _ANSWER_QUOTE.sub(_repl, text or "")
    if removed:
        cleaned = _ATTR_DEBRIS.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned).strip()
    return cleaned, removed


def _looks_broken(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return True
    if text.count(",") > 0 and len(re.findall(r"[а-яёa-z]{3,}", text, re.I)) < 5:
        return True
    if re.search(r"\s[,;]\s*[,;]", text):
        return True
    return False


def apply_quote_postcheck(
    *,
    text: str,
    candidates: list[QuoteCandidate],
    config: QualityPostcheckConfig,
    chunk_texts: dict[str, str] | None = None,
) -> QuotePostcheckResult:
    """Ground quotes against allowlist; scrub paths; strip or template on failure."""
    result = QuotePostcheckResult(text=text, quote_allowlist_present=bool(candidates))
    working = text or ""
    try:
        # Expand grounding surface: candidate text plus full chunk if provided.
        expanded = list(candidates)
        if chunk_texts:
            for cand in candidates:
                full = chunk_texts.get(cand.chunk_id)
                if full and full != cand.text:
                    expanded.append(
                        QuoteCandidate(
                            text=full,
                            chunk_id=cand.chunk_id,
                            source_id=cand.source_id,
                            meta=cand.meta,
                        )
                    )

        if config.path_scrubber_enabled:
            working, scrubbed = scrub_path_leaks(working)
            result.path_leak_scrubbed = scrubbed
            if scrubbed:
                result.codes.append("path_leak_scrubbed")

        attr_codes = check_critical_attribution(working, candidates)
        if attr_codes:
            result.critical_attribution_hallucination = True
            result.codes.extend(attr_codes)

        spans = _answer_spans(working)
        ungrounded = [s for s in spans if not _grounded_in_candidates(s, expanded)]
        if ungrounded or attr_codes:
            working, removed = strip_ungrounded_quotes(working, candidates=expanded)
            result.quote_removed = removed or bool(attr_codes)
            if result.quote_removed:
                result.codes.append("quote_removed")
            # Strip fabricated volume/page numbers after removal.
            if attr_codes:
                working = _ATTR_VOLUME.sub("", working)
                working = _ATTR_PAGE.sub("", working)
                working = _ATTR_WORK.sub("", working)
                working = _LENIN_WROTE.sub("", working)
                working = re.sub(r"\s{2,}", " ", working).strip()

        soft = str(getattr(config, "quote_postcheck_enforce_mode", "soft") or "soft") == "soft"
        min_tokens = int(getattr(config, "min_recoverable_tokens", 8) or 8)
        token_count = len(re.findall(r"[а-яёa-z0-9]{2,}", working, flags=re.IGNORECASE))
        unrecoverable = _looks_broken(working) and token_count < min_tokens
        if unrecoverable and not soft:
            working = config.static_safe_template
            result.used_static_template = True
            result.codes.append("static_safe_template")
        elif unrecoverable and soft:
            # Soft mode: keep stripped body when any recoverable tokens remain.
            result.codes.append("soft_keep_stripped_body")
            result.metadata["quote_repair_applied"] = True
        elif result.quote_removed:
            result.metadata["quote_repair_applied"] = True

        result.text = working
        result.metadata = {
            **result.metadata,
            "quoted_spans": len(spans),
            "ungrounded_spans": len(ungrounded),
            "allowlist_size": len(candidates),
            "repair_success": bool(working.strip()) and not result.used_static_template,
        }
    except (TypeError, ValueError, re.error) as exc:
        logger.exception("quote_postcheck_failed")
        # Parser failures remain the only hard-template path in soft mode.
        result.text = config.static_safe_template
        result.used_static_template = True
        result.codes.append(f"quote_postcheck_error:{type(exc).__name__}")
    return result

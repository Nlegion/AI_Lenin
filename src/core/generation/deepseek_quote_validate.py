"""DeepSeek quote validation and provider-local cleanup."""

from __future__ import annotations

import re

from src.core.generation.deepseek_text_repair import repair_strip_holes
from src.core.generation.quote_allowlist import QuoteCandidate
from src.core.generation.text_normalize import normalize_for_grounding

# Same grammar as quote_postcheck._ANSWER_QUOTE (duplicated to avoid coupling).
_ANSWER_QUOTE = re.compile(
    r"«([^»]{3,400})»|"
    r"\"([^\"]{3,400})\"|"
    r"„([^“]{3,400})“|"
    r"“([^”]{3,400})”"
)
_NO_QUOTE_PHRASE = re.compile(
    r"в\s+предоставленном\s+контексте\s+подходящей\s+цитаты\s+нет",
    re.IGNORECASE,
)
_ATTR_WITHOUT_QUOTE = re.compile(
    r"(?ix)"
    r"(?:"
    r"как\s+(?:отмечал|писал|говорил|подчёркивал|подчеркивал)\s+"
    r"(?:ленин|он)?\s*[,:—\-]?\s*"
    r"|ленин\s+(?:отмечал|писал|говорил|подчёркивал|подчеркивал)\s*"
    r"(?:что\s*)?"
    r"|по\s+словам\s+(?:ленина)?\s*[,:—\-]?\s*"
    r"|в\s+работе\s+[^\s,.!?]{0,40}\s*[,:—\-]?\s*"
    r")"
    r"(?![«\"„“])"
)
_UNCLOSED_QUOTE = re.compile(r"[«\"„“][^»\"”\n]{0,400}(?=$|\n|Механизм|Вывод|Факт)")
_ORPHAN_CLOSE_QUOTE = re.compile(r"(?<![«\"„“])([^\n«\"„“]{0,80}[»\"”])")
_EMPTY_DASH_CLAUSE = re.compile(r"\s*[—–-]\s*(?=[,.!?]|$|\n)")


def extract_answer_quote_spans(text: str) -> list[str]:
    spans: list[str] = []
    for match in _ANSWER_QUOTE.finditer(text or ""):
        span = next((g for g in match.groups() if g), None)
        if span:
            spans.append(span.strip())
    return spans


def has_no_quote_phrase(text: str) -> bool:
    return bool(_NO_QUOTE_PHRASE.search(text or ""))


def has_no_quote_conflict(text: str) -> bool:
    """True when model claims no quote but still emits quote spans."""
    if not has_no_quote_phrase(text):
        return False
    return bool(extract_answer_quote_spans(text))


def scrub_quote_debris(text: str) -> str:
    """Remove orphan attribution and unclosed quote fragments after strip."""
    cleaned = text or ""
    cleaned = _UNCLOSED_QUOTE.sub("", cleaned)
    while cleaned.count("»") > cleaned.count("«") or cleaned.count('"') % 2 == 1:
        updated = _ORPHAN_CLOSE_QUOTE.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated
    for _ in range(3):
        updated = _ATTR_WITHOUT_QUOTE.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = _EMPTY_DASH_CLAUSE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"(?i)\bленин\s*,\s*", "", cleaned)
    cleaned = re.sub(r":\s*\.", ".", cleaned)
    return cleaned.strip()


def quote_grounded_in_excerpts(
    *,
    text: str,
    excerpts: list[QuoteCandidate],
) -> bool:
    if not excerpts:
        return False
    excerpt_norms = [
        normalize_for_grounding(item.text) for item in excerpts if item.text
    ]
    excerpt_norms = [n for n in excerpt_norms if n]
    if not excerpt_norms:
        return False
    for span in extract_answer_quote_spans(text):
        norm = normalize_for_grounding(span)
        if not norm:
            continue
        for candidate in excerpt_norms:
            if norm in candidate:
                return True
    return False


def deepseek_raw_quote_ok(
    *,
    text: str,
    excerpts: list[QuoteCandidate],
    usable_excerpts: bool,
) -> bool:
    """Whether raw text satisfies DeepSeek quote policy before shared postcheck."""
    if has_no_quote_conflict(text):
        return False
    if not usable_excerpts:
        return not extract_answer_quote_spans(text) or quote_grounded_in_excerpts(
            text=text,
            excerpts=excerpts,
        )
    return quote_grounded_in_excerpts(text=text, excerpts=excerpts)


def strip_all_quote_spans(text: str) -> str:
    cleaned = _ANSWER_QUOTE.sub("", text or "")
    return scrub_quote_debris(cleaned)


def keep_only_grounded_quotes(
    *,
    text: str,
    excerpts: list[QuoteCandidate],
) -> str:
    """Drop ungrounded quote spans; keep grounded ones intact."""
    if has_no_quote_phrase(text):
        return strip_all_quote_spans(text)

    excerpt_norms = [
        normalize_for_grounding(item.text) for item in excerpts if item.text
    ]
    excerpt_norms = [n for n in excerpt_norms if n]

    def _repl(match: re.Match[str]) -> str:
        span = next((g for g in match.groups() if g), "")
        if not span:
            return ""
        norm = normalize_for_grounding(span)
        if not norm:
            return ""
        for candidate in excerpt_norms:
            if norm in candidate:
                return match.group(0)
        return ""

    cleaned = _ANSWER_QUOTE.sub(_repl, text or "")
    return scrub_quote_debris(cleaned)


def finalize_deepseek_quotes(
    *,
    text: str,
    excerpts: list[QuoteCandidate],
    usable_excerpts: bool,
) -> tuple[str, dict[str, bool]]:
    """Provider-local cleanup after shared postcheck."""
    flags = {
        "deepseek_no_quote_conflict": has_no_quote_conflict(text),
        "deepseek_stripped_all_quotes": False,
        "deepseek_stripped_ungrounded": False,
        "deepseek_scrubbed_debris": False,
        "deepseek_repaired_holes": False,
    }
    working = text
    if has_no_quote_phrase(working) or not usable_excerpts:
        before = working
        working = strip_all_quote_spans(working)
        flags["deepseek_stripped_all_quotes"] = before != working
    else:
        before = working
        working = keep_only_grounded_quotes(text=working, excerpts=excerpts)
        flags["deepseek_stripped_ungrounded"] = before != working
    after_scrub = scrub_quote_debris(working)
    repaired = repair_strip_holes(after_scrub)
    flags["deepseek_scrubbed_debris"] = after_scrub != working
    flags["deepseek_repaired_holes"] = repaired != after_scrub
    return repaired, flags

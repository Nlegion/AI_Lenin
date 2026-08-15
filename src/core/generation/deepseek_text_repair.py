"""Repair sentence holes left after quote strip / shared postcheck (DeepSeek-only)."""

from __future__ import annotations

import re

_STRUCTURE = re.compile(
    r"(?i)^\s*(факт|механизм|вывод)\s*:\s*",
)
_PUNCT_COLLISION = re.compile(r"\s*[:;,]\s*([:;.])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_EMPTY_AFTER_COLON = re.compile(r":\s*(?=[.!?]|$)")
_HANGING_PREP = re.compile(
    r"(?i)\b(?:в|во|на|по|для|из|от|до|без|при|через|между|над|под)\s*[.!?…]\s*$"
)
_STUMP_WORD = re.compile(r"(?i)\b[а-яёa-z]{1,2}\s*[.!?…]")
_DANGLING_DASH_CLAUSE = re.compile(r"—\s*[А-ЯЁA-Z][а-яёa-z]{0,20}\s*$")
_LOW_START = re.compile(
    r"(?i)^(и|а|но|или|что|как|когда|где|который|которая|которые)\b"
)
_CONTENT = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)


def _content_token_count(text: str) -> int:
    return len(_CONTENT.findall(text or ""))


_BLOCK_SPLIT = re.compile(
    r"(?=(?:^|\n)\s*(?:Факт|Механизм|Вывод)\s*:)",
    re.IGNORECASE | re.MULTILINE,
)


def _split_blocks(text: str) -> list[str]:
    """Split preserving structure labels as block starts."""
    raw = (text or "").strip()
    if not raw:
        return []
    parts = _BLOCK_SPLIT.split(raw)
    return [part.strip() for part in parts if part and part.strip()]


def _split_sentences(block: str) -> list[str]:
    label_match = _STRUCTURE.match(block)
    label = ""
    body = block
    if label_match:
        label = label_match.group(0)
        body = block[label_match.end() :]
    sentences = re.split(r"(?<=[.!?…])\s+", body.strip()) if body.strip() else []
    sentences = [item.strip() for item in sentences if item.strip()]
    if not sentences and body.strip():
        sentences = [body.strip()]
    if label and sentences:
        sentences[0] = f"{label}{sentences[0]}"
    elif label and not sentences:
        sentences = [label.rstrip(": ").strip() + "."]
    return sentences


def _sentence_ok(sentence: str) -> bool:
    working = sentence.strip()
    if not working:
        return False
    label_match = _STRUCTURE.match(working)
    body = working[label_match.end() :].strip() if label_match else working
    if not body:
        return False
    if _content_token_count(body) < 5:
        return False
    if _HANGING_PREP.search(body):
        return False
    if _STUMP_WORD.search(body):
        return False
    if _DANGLING_DASH_CLAUSE.search(body):
        return False
    if re.search(r"[:;,]\s*$", body):
        return False
    # Incomplete glue after a stripped quote, e.g. "описанный Равнодушие".
    if re.search(r"(?i)\b(?:описанн\w*|названн\w*|указанн\w*)\s+[А-ЯЁA-Z]", body):
        return False
    if _LOW_START.match(body) and _content_token_count(body) < 12:
        return False
    if re.search(r"(?i)\b(?:указан[ао]?|прямо указан[ао]?)\s*$", body.rstrip(".")):
        return False
    return True


def repair_strip_holes(text: str) -> str:
    """Fix punctuation holes and drop broken remnant sentences after quote strip."""
    cleaned = text or ""
    cleaned = _PUNCT_COLLISION.sub(r"\1", cleaned)
    cleaned = _EMPTY_AFTER_COLON.sub(".", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)

    kept: list[str] = []
    for block in _split_blocks(cleaned):
        for sentence in _split_sentences(block):
            fixed = _PUNCT_COLLISION.sub(r"\1", sentence)
            fixed = _EMPTY_AFTER_COLON.sub(".", fixed)
            fixed = _SPACE_BEFORE_PUNCT.sub(r"\1", fixed)
            fixed = _MULTI_SPACE.sub(" ", fixed).strip()
            if _sentence_ok(fixed):
                kept.append(fixed)
    if not kept:
        return cleaned.strip()
    return "\n\n".join(kept).strip()

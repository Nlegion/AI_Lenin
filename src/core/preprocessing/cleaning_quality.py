"""Quality metrics for cleaned corpus validation."""

from __future__ import annotations

import re


def split_paragraphs(text: str, min_chars: int) -> list[str]:
    chunks = [item.strip() for item in text.split("\n\n")]
    return [item for item in chunks if len(item) >= min_chars]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", text.lower()))


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def semantic_damage_ratio(
    original_text: str,
    cleaned_text: str,
    min_paragraph_chars: int,
    overlap_threshold: float,
) -> float:
    original_paragraphs = split_paragraphs(
        text=original_text, min_chars=min_paragraph_chars
    )
    cleaned_paragraphs = split_paragraphs(
        text=cleaned_text, min_chars=min_paragraph_chars
    )
    if not original_paragraphs:
        return 0.0
    if not cleaned_paragraphs:
        return 1.0

    preserved = 0
    for paragraph in original_paragraphs:
        best_match = max(
            _overlap_score(paragraph, candidate) for candidate in cleaned_paragraphs
        )
        if best_match >= overlap_threshold:
            preserved += 1

    damage = 1 - (preserved / len(original_paragraphs))
    return max(0.0, min(1.0, damage))

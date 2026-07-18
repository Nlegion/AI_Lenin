"""Deterministic text cleaner for philosophical corpus rebuilding."""

from __future__ import annotations

import re

from src.core.preprocessing.cleaning_config import CleaningConfig


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_content_start(lines: list[str], markers: list[str]) -> int:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if any(re.match(marker, stripped, flags=re.IGNORECASE) for marker in markers):
            return index
    for index, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) >= 120 and re.search(r"[а-яА-ЯёЁ]{4,}", stripped):
            return index
    return 0


def _remove_line_noise(lines: list[str], patterns: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns):
            continue
        cleaned.append(line)
    return cleaned


def _remove_inline_noise(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_document(text: str, config: CleaningConfig) -> str:
    normalized = normalize_text(text=text)
    lines = normalized.split("\n")
    start_index = _find_content_start(lines=lines, markers=config.content_start_markers)
    content_lines = lines[start_index:]
    content_lines = _remove_line_noise(lines=content_lines, patterns=config.remove_line_patterns)
    cleaned = "\n".join(content_lines)
    cleaned = _remove_inline_noise(text=cleaned, patterns=config.remove_inline_patterns)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()

"""Chunking v2 for philosophical discourse with hierarchy metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from src.core.preprocessing.chunking_config import ChunkingConfig


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    source_path: str
    author: str
    work: str
    stance_type: str
    chapter: str
    section: str
    paragraph_index: int
    thesis_index: int
    chunk_index: int
    token_count: int
    char_start: int
    char_end: int
    text: str
    boundary_ok: bool


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    return paragraphs


def _detect_header(line: str, markers: list[str]) -> str | None:
    for marker in markers:
        if re.match(marker, line.strip(), flags=re.IGNORECASE):
            return line.strip()
    return None


def _split_theses(paragraph: str, markers: list[str]) -> list[str]:
    if not paragraph:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+", paragraph.strip())
    theses: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer} {part}".strip() if buffer else part.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        has_marker = any(marker.lower() in lowered for marker in markers)
        if has_marker and buffer:
            theses.append(buffer.strip())
            buffer = part.strip()
        else:
            buffer = candidate
    if buffer:
        theses.append(buffer.strip())
    return theses or [paragraph.strip()]


def _build_chunk_id(
    source_id: str, chunk_index: int, char_start: int, char_end: int
) -> str:
    base = f"{source_id}:{chunk_index}:{char_start}:{char_end}"
    digest = hashlib.sha1(base.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]
    return f"chunk_{digest}"


def _boundary_ok(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped and stripped[-1] in ".!?;:")


def _split_tokens_with_overlap(
    tokens: list[str], max_tokens: int, overlap_tokens: int
) -> list[list[str]]:
    if len(tokens) <= max_tokens:
        return [tokens]
    result: list[list[str]] = []
    step = max(1, max_tokens - overlap_tokens)
    start = 0
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        if not window:
            break
        result.append(window)
        if start + max_tokens >= len(tokens):
            break
        start += step
    return result


def chunk_document(
    *,
    source_id: str,
    source_path: str,
    author: str,
    work: str,
    stance_type: str,
    text: str,
    config: ChunkingConfig,
) -> list[ChunkRecord]:
    paragraphs = _split_paragraphs(text=text)
    chunks: list[ChunkRecord] = []
    current_chapter = "unknown"
    current_section = "unknown"
    chunk_index = 0
    char_cursor = 0

    max_tokens = max(config.max_tokens, config.min_tokens)
    overlap_tokens = max(1, int(max_tokens * config.overlap_ratio))
    token_buffer: list[str] = []
    text_buffer = ""
    source_start = 0
    paragraph_index = -1
    thesis_index = -1

    def flush_buffer(boundary: bool, force: bool = False) -> None:
        nonlocal chunk_index, token_buffer, text_buffer, source_start
        if not text_buffer.strip():
            return
        token_count = len(tokenize(text_buffer))
        if token_count == 0 or len(text_buffer.strip()) < config.min_chunk_chars:
            return
        if token_count < config.min_tokens and not force:
            return
        char_end = source_start + len(text_buffer)
        chunk_id = _build_chunk_id(
            source_id=source_id,
            chunk_index=chunk_index,
            char_start=source_start,
            char_end=char_end,
        )
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                source_id=source_id,
                source_path=source_path,
                author=author,
                work=work,
                stance_type=stance_type,
                chapter=current_chapter,
                section=current_section,
                paragraph_index=paragraph_index,
                thesis_index=thesis_index,
                chunk_index=chunk_index,
                token_count=token_count,
                char_start=source_start,
                char_end=char_end,
                text=text_buffer.strip(),
                boundary_ok=boundary
                or token_count >= int(config.max_tokens * 0.95)
                or _boundary_ok(text_buffer),
            )
        )
        chunk_index += 1
        overlap = token_buffer[-overlap_tokens:] if token_buffer else []
        token_buffer = overlap.copy()
        text_buffer = " ".join(overlap).strip()
        source_start = max(0, char_end - len(text_buffer))

    for paragraph_index, paragraph in enumerate(paragraphs):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if lines:
            chapter_match = _detect_header(lines[0], config.chapter_markers)
            section_match = _detect_header(lines[0], config.section_markers)
            if chapter_match:
                current_chapter = chapter_match
            if section_match:
                current_section = section_match

        theses = _split_theses(paragraph=paragraph, markers=config.thesis_markers)
        for thesis_index, thesis in enumerate(theses):
            thesis_tokens = tokenize(thesis)
            if not thesis_tokens:
                char_cursor += len(thesis) + 1
                continue

            thesis_segments = _split_tokens_with_overlap(
                tokens=thesis_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )

            if not token_buffer:
                source_start = char_cursor

            for segment_idx, segment in enumerate(thesis_segments):
                remaining = segment[:]
                is_last_segment = segment_idx == len(thesis_segments) - 1
                while remaining:
                    if not token_buffer:
                        source_start = char_cursor
                    available = max_tokens - len(token_buffer)
                    if available <= 0:
                        flush_buffer(boundary=False, force=True)
                        continue
                    taken = remaining[:available]
                    remaining = remaining[available:]
                    token_buffer.extend(taken)
                    text_buffer = f"{text_buffer} {' '.join(taken)}".strip()
                    if len(token_buffer) >= max_tokens:
                        flush_buffer(
                            boundary=is_last_segment and not remaining, force=True
                        )

            char_cursor += len(thesis) + 1

        char_cursor += 1

    flush_buffer(boundary=True, force=True)
    return chunks

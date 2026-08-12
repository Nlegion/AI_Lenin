"""Human-readable txt formatting for quality / live QA artifacts."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts._quality_qa_io import QaItem

_HALLUCINATION_PREFIX_RE = re.compile(
    r"(?:^|\s)В стилизованной интерпретации\s*:\s*",
    flags=re.IGNORECASE,
)
_SECTION_SPLIT_RE = re.compile(
    r"(?=(?:\*{0,2}(?:Факт|Механизм|Вывод)\*{0,2}\s*:))",
    flags=re.IGNORECASE,
)
_DISCLAIMER_RE = re.compile(
    r"(Ответ сгенерирован ИИ\b.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)
_SECTION_LABEL_RE = re.compile(
    r"^(\*{0,2})(Факт|Механизм|Вывод)(\*{0,2})\s*:\s*",
    flags=re.IGNORECASE,
)
_SECTION_CANON = {"факт": "Факт", "механизм": "Механизм", "вывод": "Вывод"}


def format_txt_header() -> str:
    # Human-readable txt artifacts contain answer body only (no Q/context).
    return ""


def format_answer_for_display(answer: str) -> str:
    """Keep Факт / Механизм / Вывод / disclaimer; drop prompt/context chrome."""
    text = str(answer or "").strip()
    if not text:
        return ""

    text = _HALLUCINATION_PREFIX_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    disclaimer = ""
    disclaimer_match = _DISCLAIMER_RE.search(text)
    if disclaimer_match is not None:
        disclaimer = disclaimer_match.group(1).strip()
        text = text[: disclaimer_match.start()].strip()

    chunks = [part.strip() for part in _SECTION_SPLIT_RE.split(text) if part.strip()]
    sections: list[str] = []
    for chunk in chunks:
        label_match = _SECTION_LABEL_RE.match(chunk)
        if label_match is None:
            if chunk:
                sections.append(chunk)
            continue
        canon = _SECTION_CANON[label_match.group(2).casefold()]
        body = chunk[label_match.end() :].strip()
        sections.append(f"{canon} : {body}" if body else f"{canon} :")

    if not sections and text:
        sections = [text]
    if disclaimer:
        sections.append(disclaimer)
    return "\n\n".join(sections).strip()


def format_txt_block(
    *,
    index: int,
    item: QaItem,
    answer: str,
    txt_max_chars: int = 0,
) -> str:
    body = format_answer_for_display(answer)
    if txt_max_chars > 0 and len(body) > txt_max_chars:
        body = body[:txt_max_chars].rstrip() + "\n...[truncated]"
    topic = item.topic or "n/a"
    return f"=== {index} / {item.id} [{topic}] ===\n{body}\n\n"

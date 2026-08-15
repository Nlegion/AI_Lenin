"""Query rewriting, decomposition, and HyDE helpers."""

from __future__ import annotations

import re


IDEOLOGY_TERMS = [
    ("санкции", "империалистические экономические санкции"),
    ("инфляция", "кризис капиталистического воспроизводства"),
    ("безработица", "резервная армия труда"),
    ("приватизация", "передача общественного богатства частному капиталу"),
    ("монополия", "монополистический капитал"),
    ("война", "империалистический конфликт"),
    ("выборы", "буржуазная парламентская политика"),
]


def rewrite_query_to_philosophical_register(text: str) -> str:
    rewritten = text
    lowered = text.lower()
    for needle, replacement in IDEOLOGY_TERMS:
        if needle in lowered:
            rewritten = f"{rewritten}. {replacement}"
    return rewritten.strip()


def decompose_query(text: str) -> tuple[str, str]:
    sentence_parts = [item.strip() for item in re.split(r"[.!?]", text) if item.strip()]
    if not sentence_parts:
        return text, text

    fact_markers = ("кто", "что", "где", "когда", "сколько", "произош")
    fact_lines: list[str] = []
    evaluative_lines: list[str] = []
    for line in sentence_parts:
        lowered = line.lower()
        if any(marker in lowered for marker in fact_markers) or re.search(
            r"\d", lowered
        ):
            fact_lines.append(line)
        else:
            evaluative_lines.append(line)

    factual = ". ".join(fact_lines).strip() or sentence_parts[0]
    evaluative = ". ".join(evaluative_lines).strip() or sentence_parts[-1]
    return factual, evaluative


def build_hyde_query(text: str) -> str:
    return (
        "Гипотетический ленинский тезис: определить классовые интересы, "
        "материальную основу конфликта, форму империалистического давления и "
        "возможные противоречия. Тема: "
        f"{text}"
    )

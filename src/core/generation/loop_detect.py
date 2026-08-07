"""Near-duplicate paragraph loop detection and cheap fixes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.core.generation.text_normalize import normalize_for_grounding
from src.core.settings.quality_postcheck_config import LoopConfig, QualityPostcheckConfig

_PARA_SPLIT = re.compile(r"\n{2,}|(?<=[.!?…])\s+(?=[А-ЯЁA-Z«\"])")
_TOKEN = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


@dataclass
class LoopFixResult:
    text: str
    loop_detected: bool = False
    loop_action: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(normalize_for_grounding(text)) if len(t) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _paragraphs(text: str, *, min_chars: int) -> list[str]:
    parts = [p.strip() for p in _PARA_SPLIT.split(text.strip()) if p.strip()]
    if len(parts) <= 1:
        # Fallback: sentence-like chunks for single-block loops.
        parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]
    return [p for p in parts if len(p) >= min_chars] or ([text.strip()] if text.strip() else [])


def detect_and_fix_loops(
    text: str,
    *,
    config: QualityPostcheckConfig,
    rag_empty: bool = False,
) -> LoopFixResult:
    if not config.loop_fix_enabled or not text.strip():
        return LoopFixResult(text=text)
    loop_cfg: LoopConfig = config.loop
    paras = _paragraphs(text, min_chars=loop_cfg.min_paragraph_chars)
    if len(paras) < 2:
        return LoopFixResult(text=text)

    tokenized = [_tokens(p) for p in paras]
    drop_indices: set[int] = set()
    for i in range(1, len(paras)):
        for j in range(i):
            if j in drop_indices:
                continue
            score = _jaccard(tokenized[i], tokenized[j])
            if score >= loop_cfg.jaccard_threshold:
                drop_indices.add(i)
                break

    if not drop_indices:
        return LoopFixResult(text=text)

    # Prefer dedupe of repeated paragraphs; avoid static_insufficient for benign loops.
    kept = [p for idx, p in enumerate(paras) if idx not in drop_indices]
    fixed = "\n\n".join(kept) if "\n\n" in text else " ".join(kept)
    fixed = fixed.strip()
    hard = bool(getattr(config, "hard_fallback_on_broken_output", False))
    if not fixed and hard:
        fixed = config.static_safe_template
        action = "static_safe"
    elif not fixed:
        fixed = kept[0] if kept else text
        action = "drop_duplicate_paragraph"
    else:
        action = "drop_duplicate_paragraph"
    return LoopFixResult(
        text=fixed,
        loop_detected=True,
        loop_action=action,
        metadata={
            "dropped_paragraphs": len(drop_indices),
            "rag_empty": rag_empty,
        },
    )

"""Context-bound combat detection (co-occurrence window)."""

from __future__ import annotations

import re

DEFAULT_WINDOW = 10

DEFAULT_COMBAT_STEMS = (
    "удар",
    "обстрел",
    "авиабомб",
    "фронт",
    "всу",
    "вс рф",
    "поразил",
    "ракетн",
    "миномёт",
    "миномет",
)

# Strong military co-tokens (exclude bare «арми» alone — FP «армия потребителей»).
DEFAULT_CO_TOKENS = (
    "военн",
    "обстрел",
    "войск",
    "бомб",
    "ракет",
    "фронт",
    "снп",
    "миномёт",
    "миномет",
    "артиллер",
    "авиаудар",
    "огневой",
    "всу",
    "подразделени",
)

METAPHOR_BLOCKERS = ("потребител", "покупател", "болельщик", "фанат")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"[а-яёa-z0-9]+", text.lower())]


def combat_cooccurrence_hit(
    text: str,
    *,
    combat_stems: list[str] | None = None,
    co_tokens: list[str] | None = None,
    window: int = DEFAULT_WINDOW,
) -> list[str]:
    """Combat stem near military co-token within ±window tokens → reason codes."""
    lowered = text.lower()
    if any(b in lowered for b in METAPHOR_BLOCKERS) and "военн" not in lowered and "войск" not in lowered:
        # Soft metaphor path: still allow explicit force phrases below.
        pass
    else:
        pass

    stems = [s.lower() for s in (combat_stems or list(DEFAULT_COMBAT_STEMS))]
    cos = [c.lower() for c in (co_tokens or list(DEFAULT_CO_TOKENS))]
    # Drop weak «арми» from configured lists unless paired with военн in text.
    cos = [c for c in cos if c not in {"арми", "армия", "вс"} or "военн" in lowered]

    tokens = _tokenize(text)
    combat_idx = [
        i
        for i, tok in enumerate(tokens)
        if any(tok.startswith(stem) for stem in stems if " " not in stem)
    ]
    co_idx = [
        i
        for i, tok in enumerate(tokens)
        if any(tok.startswith(co) or (len(co) >= 5 and co in tok) for co in cos)
    ]

    hits: list[str] = []
    for stem in stems:
        if " " in stem and stem in lowered:
            hits.append(f"combat:{stem}")

    metaphor = any(b in lowered for b in METAPHOR_BLOCKERS)
    if not metaphor:
        for ci in combat_idx:
            for oi in co_idx:
                if ci != oi and abs(ci - oi) <= window:
                    hits.append(f"combat_co:{tokens[ci]}+{tokens[oi]}")
                    return hits

    for phrase in (
        "вс рф",
        "вооруженные силы рф",
        "вооружённые силы рф",
        "боевые действия",
        "военных подразделениях рф",
        "армии россии",
    ):
        if phrase in lowered:
            hits.append(f"military_phrase:{phrase}")
    return hits


def military_rf_context_hit(text: str) -> bool:
    patterns = [
        r"(военн\w+|арм\w+|силов\w+).{0,40}(рф|росси\w+)",
        r"(рф|росси\w+).{0,40}(военн\w+|арм\w+|силов\w+)",
    ]
    lowered = text.lower()
    if any(b in lowered for b in METAPHOR_BLOCKERS):
        return False
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)

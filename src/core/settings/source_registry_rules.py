"""Default source registry classification rules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceRegistryRules:
    """Typed rules for classifying corpus sources."""

    core_authors: set[str] = field(
        default_factory=lambda: {
            "ленин",
            "владимир ленин",
            "в.и.ленин",
            "в и ленин",
            "pss",
            "single",
        }
    )
    influence_agree_authors: set[str] = field(
        default_factory=lambda: {
            "маркс",
            "энгельс",
            "марксэнгельс",
            "карл маркс",
            "фридрих энгельс",
            "marx",
            "engels",
        }
    )
    influence_critical_authors: set[str] = field(
        default_factory=lambda: {
            "бернштейн",
            "каутский",
            "меньшевики",
            "ревизионизм",
        }
    )
    contextual_authors: set[str] = field(default_factory=set)
    path_overrides: dict[str, str] = field(default_factory=dict)
    allowed_extensions: tuple[str, ...] = (".txt", ".md")


DEFAULT_SOURCE_REGISTRY_RULES = SourceRegistryRules()

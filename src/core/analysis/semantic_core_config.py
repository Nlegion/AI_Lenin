"""Semantic core config loader with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

import yaml

logger = logging.getLogger(__name__)

_CHARSET_BOUNDARY = r"(?<![а-яёa-z0-9]){token}(?![а-яёa-z0-9])"
_STEM_BOUNDARY = r"(?<![а-яёa-z0-9]){token}[а-яё]*(?![а-яёa-z0-9])"


@dataclass(frozen=True)
class TriggerSpec:
    text: str
    pattern: re.Pattern[str] = field(repr=False, compare=False)
    weight: float = 1.0
    match: str = "phrase"


@dataclass(frozen=True)
class AbstractTopic:
    topic_id: str
    label: str
    synthesis_hint: str
    hint_only: bool
    triggers: tuple[TriggerSpec, ...]
    retrieval_terms: tuple[str, ...]


@dataclass(frozen=True)
class SemanticCoreConfig:
    enabled: bool = False
    apply_to_dialectical: bool = True
    apply_to_legacy: bool = False
    include_axes_in_semantic_query: bool = False
    include_title_anchor: bool = False
    empty_r1_fallback_to_legacy_slot_query: bool = True
    max_topics_logged: int = 3
    max_terms_per_topic: int = 3
    max_term_tokens: int = 5
    max_term_tokens_enforcement: str = "warn"
    max_query_chars: int = 512
    max_title_anchor_chars: int = 120
    baseline_query_content_tokens: int = 5
    phase0_score_floor_ratio: float = 0.8
    author_known_rate_min: float = 0.6
    cliche_warn_rate_max_ratio: float = 1.5
    cliche_warn_rate_min_delta_pp: float = 5.0
    normalize_yo_for_routing: bool = True
    allow_duplicate_triggers: bool = False
    multi_topic_file_audit: bool = False
    embedder_model_path: str = "models/Giga-Embeddings-instruct"
    embedder_model_max_tokens: int | None = None
    embedder_token_margin: int = 32
    lacuna_hedge_gate: str = "warn_only"
    lenin_author_aliases: tuple[str, ...] = ()
    author_reject_substrings: tuple[str, ...] = ()
    retrieval_term_stopwords: frozenset[str] = field(default_factory=frozenset)
    topics: tuple[AbstractTopic, ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compile_trigger(text: str, match: str) -> re.Pattern[str]:
    lowered = text.casefold()
    if match == "charset_boundary":
        token = re.escape(lowered)
        return re.compile(_CHARSET_BOUNDARY.format(token=token), re.IGNORECASE)
    if match == "stem":
        token = re.escape(lowered)
        return re.compile(_STEM_BOUNDARY.format(token=token), re.IGNORECASE)
    parts = [re.escape(part) for part in lowered.split() if part]
    if not parts:
        raise ValueError(f"empty trigger text: {text!r}")
    if len(parts) == 1:
        body = parts[0]
    else:
        body = r"\s+".join(parts)
    return re.compile(
        _CHARSET_BOUNDARY.format(token=body),
        re.IGNORECASE,
    )


def _parse_trigger(raw: object) -> TriggerSpec:
    if isinstance(raw, str):
        text = raw.strip()
        match = "phrase"
        weight = 1.0
    elif isinstance(raw, dict):
        text = str(raw.get("text", "")).strip()
        match = str(raw.get("match", "phrase")).strip() or "phrase"
        weight = float(raw.get("weight", 1.0))
    else:
        raise ValueError(f"invalid trigger: {raw!r}")
    if not text:
        raise ValueError("trigger text must be non-empty")
    if weight <= 0:
        raise ValueError(f"trigger weight must be > 0 for {text!r}")
    if match not in {"phrase", "charset_boundary", "stem"}:
        raise ValueError(f"unsupported trigger match: {match}")
    return TriggerSpec(
        text=text,
        pattern=_compile_trigger(text=text, match=match),
        weight=weight,
        match=match,
    )


def _validate_retrieval_term(
    term: str,
    *,
    stopwords: frozenset[str],
    max_term_tokens: int,
    enforcement: str,
    topic_id: str,
) -> None:
    tokens = [token for token in term.casefold().split() if token]
    if not tokens:
        raise ValueError(f"empty retrieval_term in topic {topic_id}")
    if not any(token not in stopwords and len(token) > 1 for token in tokens):
        raise ValueError(
            f"retrieval_term is stopword-only in topic {topic_id}: {term!r}"
        )
    if len(tokens) > max_term_tokens:
        message = (
            f"retrieval_term exceeds max_term_tokens={max_term_tokens} "
            f"in topic {topic_id}: {term!r}"
        )
        if enforcement == "fail":
            raise ValueError(message)
        logger.warning(message)


def load_semantic_core_config(path: Path | None = None) -> SemanticCoreConfig:
    config_path = path or (_repo_root() / "config" / "semantic_core.yaml")
    if not config_path.exists():
        return SemanticCoreConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("semantic_core") or {}
    if not isinstance(section, dict):
        raise ValueError("semantic_core section must be a mapping")

    stopwords = frozenset(
        str(item).casefold()
        for item in (section.get("retrieval_term_stopwords") or [])
    )
    max_term_tokens = int(section.get("max_term_tokens", 5))
    if max_term_tokens <= 0:
        raise ValueError("max_term_tokens must be > 0")
    max_query_chars = int(section.get("max_query_chars", 512))
    if max_query_chars <= 0:
        raise ValueError("max_query_chars must be > 0")
    max_terms = int(section.get("max_terms_per_topic", 3))
    if max_terms <= 0:
        raise ValueError("max_terms_per_topic must be > 0")
    enforcement = str(section.get("max_term_tokens_enforcement", "warn"))
    allow_dup = bool(section.get("allow_duplicate_triggers", False))

    topics_raw = payload.get("abstract_topics") or {}
    if not isinstance(topics_raw, dict):
        raise ValueError("abstract_topics must be a mapping")

    seen_triggers: dict[str, str] = {}
    topics: list[AbstractTopic] = []
    for topic_id, topic_payload in topics_raw.items():
        if not isinstance(topic_payload, dict):
            raise ValueError(f"topic {topic_id} must be a mapping")
        triggers = tuple(
            _parse_trigger(item) for item in (topic_payload.get("triggers") or [])
        )
        for trigger in triggers:
            key = f"{trigger.match}:{trigger.text.casefold()}"
            if key in seen_triggers and not allow_dup:
                raise ValueError(
                    f"duplicate trigger {trigger.text!r} in "
                    f"{seen_triggers[key]} and {topic_id}"
                )
            seen_triggers[key] = topic_id

        terms = [
            str(item).strip()
            for item in (topic_payload.get("retrieval_terms") or [])
            if str(item).strip()
        ]
        hint_only = bool(topic_payload.get("hint_only", False))
        if not terms and not hint_only:
            logger.warning(
                "semantic_core_coerce_hint_only topic_id=%s",
                topic_id,
            )
            hint_only = True
        if hint_only and terms:
            logger.warning(
                "semantic_core_hint_only_strips_terms topic_id=%s",
                topic_id,
            )
            terms = []
        for term in terms:
            _validate_retrieval_term(
                term,
                stopwords=stopwords,
                max_term_tokens=max_term_tokens,
                enforcement=enforcement,
                topic_id=str(topic_id),
            )
        topics.append(
            AbstractTopic(
                topic_id=str(topic_id),
                label=str(topic_payload.get("label", topic_id)),
                synthesis_hint=str(topic_payload.get("synthesis_hint", "")),
                hint_only=hint_only,
                triggers=triggers,
                retrieval_terms=tuple(terms[:max_terms]),
            )
        )

    model_max = section.get("embedder_model_max_tokens")
    return SemanticCoreConfig(
        enabled=bool(section.get("enabled", False)),
        apply_to_dialectical=bool(section.get("apply_to_dialectical", True)),
        apply_to_legacy=bool(section.get("apply_to_legacy", False)),
        include_axes_in_semantic_query=bool(
            section.get("include_axes_in_semantic_query", False)
        ),
        include_title_anchor=bool(section.get("include_title_anchor", False)),
        empty_r1_fallback_to_legacy_slot_query=bool(
            section.get("empty_r1_fallback_to_legacy_slot_query", True)
        ),
        max_topics_logged=int(section.get("max_topics_logged", 3)),
        max_terms_per_topic=max_terms,
        max_term_tokens=max_term_tokens,
        max_term_tokens_enforcement=enforcement,
        max_query_chars=max_query_chars,
        max_title_anchor_chars=int(section.get("max_title_anchor_chars", 120)),
        baseline_query_content_tokens=int(
            section.get("baseline_query_content_tokens", 5)
        ),
        phase0_score_floor_ratio=float(section.get("phase0_score_floor_ratio", 0.8)),
        author_known_rate_min=float(section.get("author_known_rate_min", 0.6)),
        cliche_warn_rate_max_ratio=float(
            section.get("cliche_warn_rate_max_ratio", 1.5)
        ),
        cliche_warn_rate_min_delta_pp=float(
            section.get("cliche_warn_rate_min_delta_pp", 5)
        ),
        normalize_yo_for_routing=bool(section.get("normalize_yo_for_routing", True)),
        allow_duplicate_triggers=allow_dup,
        multi_topic_file_audit=bool(section.get("multi_topic_file_audit", False)),
        embedder_model_path=str(
            section.get("embedder_model_path", "models/Giga-Embeddings-instruct")
        ),
        embedder_model_max_tokens=int(model_max) if model_max is not None else None,
        embedder_token_margin=int(section.get("embedder_token_margin", 32)),
        lacuna_hedge_gate=str(section.get("lacuna_hedge_gate", "warn_only")),
        lenin_author_aliases=tuple(
            str(item) for item in (section.get("lenin_author_aliases") or [])
        ),
        author_reject_substrings=tuple(
            str(item) for item in (section.get("author_reject_substrings") or [])
        ),
        retrieval_term_stopwords=stopwords,
        topics=tuple(topics),
    )

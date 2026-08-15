"""Loader for SafetyGate feature flags and policy config."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
import yaml

logger = logging.getLogger(__name__)

EnforceMode = Literal["old", "new"]


class SafetyGateFlags(BaseModel):
    enabled: bool = False
    shadow_mode: bool = True
    enforce_mode: EnforceMode = "old"
    async_shadow: bool = False
    cache_enabled: bool = True
    cache_max_entries: int = 512
    fallback_to_news_guard_keys: bool = True


class SafetyGatePolicy(BaseModel):
    """Subset of news_guard input policy mirrored for SafetyGate SoT."""

    policy_version: str = "1.0.0"
    refusal_message: str = (
        "Анализ данной темы невозможен в соответствии с политикой безопасности."
    )
    skip_message: str = "Тема вне сферы марксистско-ленинского анализа новостей."
    classify_on_unknown_as: Literal["allow", "deny", "quarantine", "skip"] = (
        "quarantine"
    )
    economy_policy_markers: list[str] = Field(default_factory=list)
    yellow_block_patterns: list[str] = Field(default_factory=list)
    yellow_warning_text: str = (
        "Ограниченный режим анализа: экономические и политические отношения без "
        "комментариев боевых действий."
    )
    allow_topics: list[str] = Field(default_factory=list)
    hard_deny_topics: list[str] = Field(default_factory=list)
    quarantine_topics: list[str] = Field(default_factory=list)
    hard_deny_keywords: list[str] = Field(default_factory=list)
    quarantine_keywords: list[str] = Field(default_factory=list)
    military_topics: list[str] = Field(default_factory=list)
    public_interest_topics: list[str] = Field(default_factory=list)
    block_private_pii: bool = True


class SafetyGateConfig(BaseModel):
    flags: SafetyGateFlags = Field(default_factory=SafetyGateFlags)
    policy: SafetyGatePolicy = Field(default_factory=SafetyGatePolicy)
    fallback_keys_used: list[str] = Field(default_factory=list)


def default_safety_gate_config_path(base_dir: Path) -> Path:
    return base_dir / "config" / "safety_gate_config.yaml"


def _config_version_hash(payload: dict[str, Any]) -> str:
    raw = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _merge_fallback(
    section: dict[str, Any],
    *,
    news_guard_path: Path,
    allow_fallback: bool,
) -> tuple[dict[str, Any], list[str]]:
    used: list[str] = []
    if not allow_fallback or not news_guard_path.is_file():
        return section, used
    legacy = yaml.safe_load(news_guard_path.read_text(encoding="utf-8")) or {}
    ng = legacy.get("news_guard", legacy)
    input_gate = ng.get("input_gate") or {}
    policy = dict(section.get("policy") or {})
    keys = (
        "allow_topics",
        "hard_deny_topics",
        "quarantine_topics",
        "hard_deny_keywords",
        "quarantine_keywords",
        "military_topics",
        "public_interest_topics",
        "economy_policy_markers",
        "yellow_block_patterns",
        "refusal_message",
        "skip_message",
        "classify_on_unknown_as",
        "block_private_pii",
    )
    for key in keys:
        if key not in policy or policy[key] in (None, [], ""):
            if key in input_gate:
                policy[key] = input_gate[key]
                used.append(key)
                logger.warning(
                    "safety_gate_config_fallback_key",
                    extra={"key": key, "source": str(news_guard_path)},
                )
    section = {**section, "policy": policy}
    return section, used


def load_safety_gate_config(
    path: Path,
    *,
    news_guard_path: Path | None = None,
) -> SafetyGateConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload = payload or {}
    section = payload.get("safety_gate", payload)
    flags_raw = section.get("flags") or {}
    allow_fallback = bool(flags_raw.get("fallback_to_news_guard_keys", True))
    ng_path = news_guard_path or path.parent / "news_guard.yaml"
    section, used = _merge_fallback(
        section,
        news_guard_path=ng_path,
        allow_fallback=allow_fallback,
    )
    cfg = SafetyGateConfig.model_validate(section)
    cfg.fallback_keys_used = used
    return cfg


@lru_cache(maxsize=4)
def get_safety_gate_config(path_str: str) -> SafetyGateConfig:
    path = Path(path_str)
    return load_safety_gate_config(path=path)


def reload_safety_gate_config() -> None:
    get_safety_gate_config.cache_clear()


def safety_gate_version_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _config_version_hash(payload)

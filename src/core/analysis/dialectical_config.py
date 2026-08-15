"""Dialectical orchestration config loaded from retrieval_pipeline.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DialecticalOrchestrationConfig:
    enabled: bool = False
    r1_limit: int = 4
    r2_limit: int = 3
    r3_limit: int = 3
    require_r1: bool = True
    fail_on_empty_r1: bool = False
    include_axes_in_query: bool = True
    include_modality_suffix: bool = True
    axes_lemma_enabled: bool = True
    fallback_to_legacy_context: bool = False
    slot_timeout_sec: float = 3.0
    retrieve_wall_timeout_sec: float = 4.0
    widen_factor: int = 3
    max_retries: int = 2
    r1_modality_suffix: str = ""
    r2_modality_suffix: str = "поддержка"
    r3_modality_suffix: str = "критика"
    short_lead_chars: int = 200


def load_dialectical_config(config_path: Path) -> DialecticalOrchestrationConfig:
    if not config_path.exists():
        return DialecticalOrchestrationConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("dialectical_orchestration") or {}
    if not isinstance(section, dict):
        return DialecticalOrchestrationConfig()
    return DialecticalOrchestrationConfig(
        enabled=bool(section.get("enabled", False)),
        r1_limit=int(section.get("r1_limit", 4)),
        r2_limit=int(section.get("r2_limit", 3)),
        r3_limit=int(section.get("r3_limit", 3)),
        require_r1=bool(section.get("require_r1", True)),
        fail_on_empty_r1=bool(section.get("fail_on_empty_r1", False)),
        include_axes_in_query=bool(section.get("include_axes_in_query", True)),
        include_modality_suffix=bool(section.get("include_modality_suffix", True)),
        axes_lemma_enabled=bool(section.get("axes_lemma_enabled", True)),
        fallback_to_legacy_context=bool(
            section.get("fallback_to_legacy_context", False)
        ),
        slot_timeout_sec=float(section.get("slot_timeout_sec", 3.0)),
        retrieve_wall_timeout_sec=float(section.get("retrieve_wall_timeout_sec", 4.0)),
        widen_factor=int(section.get("widen_factor", 3)),
        max_retries=int(section.get("max_retries", 2)),
        r1_modality_suffix=str(section.get("r1_modality_suffix", "")),
        r2_modality_suffix=str(section.get("r2_modality_suffix", "поддержка")),
        r3_modality_suffix=str(section.get("r3_modality_suffix", "критика")),
        short_lead_chars=int(section.get("short_lead_chars", 200)),
    )

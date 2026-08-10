"""Dialectical reasoning mode and runtime config."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from src.core.dialectics.constants import MAX_RENDERED_CHARS

SCHEMA_VERSION = "1.0.0"
ENGINE_VERSION = "1.0.0"


class DialecticalMode(str, Enum):
    LEGACY = "legacy"
    ORCHESTRATION_SINGLE_PASS = "orchestration_single_pass"
    REASONING_SHADOW = "reasoning_shadow"
    REASONING_PUBLISH = "reasoning_publish"


@dataclass(frozen=True)
class DialecticalReasoningConfig:
    mode: DialecticalMode = DialecticalMode.ORCHESTRATION_SINGLE_PASS
    kill_switch: bool = False
    fixture_mode: bool = False
    max_principles: int = 6
    max_principles_per_slot: int = 3
    max_quote_chars: int = 280
    max_causal_links: int = 3
    max_error_report_chars: int = 600
    max_rendered_chars: int = MAX_RENDERED_CHARS
    max_tokens_out: int = 512
    ctx_size: int = 4096
    ctx_margin_ratio: float = 0.9
    temperature: float = 0.35
    repair_temperature: float = 0.1
    repair_max_attempts: int = 2
    per_pass_timeout_sec: float = 90.0
    global_timeout_sec: float = 180.0
    shadow_sample_rate: float = 0.1
    fallback_to_legacy_on_timeout: bool = False
    require_orchestration: bool = True
    judge_sample_rate: float = 0.0
    schema_version: str = SCHEMA_VERSION
    engine_version: str = ENGINE_VERSION


def default_retrieval_pipeline_path(base_dir: Path) -> Path:
    return base_dir / "config" / "retrieval_pipeline.yaml"


def load_dialectical_reasoning_config(
    *,
    path: Path | None = None,
    base_dir: Path | None = None,
) -> DialecticalReasoningConfig:
    config_path = path
    if config_path is None and base_dir is not None:
        config_path = default_retrieval_pipeline_path(base_dir)
    if config_path is None or not config_path.is_file():
        return DialecticalReasoningConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("dialectical_reasoning") or {}
    if not isinstance(section, dict):
        return DialecticalReasoningConfig()
    raw_mode = str(section.get("mode", DialecticalMode.ORCHESTRATION_SINGLE_PASS.value))
    try:
        mode = DialecticalMode(raw_mode)
    except ValueError:
        mode = DialecticalMode.ORCHESTRATION_SINGLE_PASS
    kill = bool(section.get("kill_switch", False))
    if kill and mode in (DialecticalMode.REASONING_SHADOW, DialecticalMode.REASONING_PUBLISH):
        mode = DialecticalMode.ORCHESTRATION_SINGLE_PASS
    return DialecticalReasoningConfig(
        mode=mode,
        kill_switch=kill,
        fixture_mode=bool(section.get("fixture_mode", False)),
        max_principles=int(section.get("max_principles", 6)),
        max_principles_per_slot=int(section.get("max_principles_per_slot", 3)),
        max_quote_chars=int(section.get("max_quote_chars", 280)),
        max_causal_links=int(section.get("max_causal_links", 3)),
        max_error_report_chars=int(section.get("max_error_report_chars", 600)),
        max_rendered_chars=min(
            int(section.get("max_rendered_chars", MAX_RENDERED_CHARS)),
            MAX_RENDERED_CHARS,
        ),
        max_tokens_out=int(section.get("max_tokens_out", 512)),
        ctx_size=int(section.get("ctx_size", 4096)),
        ctx_margin_ratio=float(section.get("ctx_margin_ratio", 0.9)),
        temperature=float(section.get("temperature", 0.35)),
        repair_temperature=float(section.get("repair_temperature", 0.1)),
        repair_max_attempts=int(section.get("repair_max_attempts", 2)),
        per_pass_timeout_sec=float(section.get("per_pass_timeout_sec", 90.0)),
        global_timeout_sec=float(section.get("global_timeout_sec", 180.0)),
        shadow_sample_rate=float(section.get("shadow_sample_rate", 0.1)),
        fallback_to_legacy_on_timeout=bool(section.get("fallback_to_legacy_on_timeout", False)),
        require_orchestration=bool(section.get("require_orchestration", True)),
        judge_sample_rate=float(section.get("judge_sample_rate", 0.0)),
    )

"""Loader for quality postcheck feature flags and thresholds."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class LoopConfig(BaseModel):
    min_paragraph_chars: int = 40
    jaccard_threshold: float = 0.85
    max_repeat_count: int = 1


class QualityPostcheckConfig(BaseModel):
    quote_allowlist_enabled: bool = True
    quote_rewrite_enabled: bool = False
    yellow_output_filter_enabled: bool = False
    loop_fix_enabled: bool = False
    loop_regen_enabled: bool = False
    path_scrubber_enabled: bool = True
    principles_fallback_enabled: bool = True
    grounded_element_check_enabled: bool = True
    hard_fallback_on_broken_output: bool = False
    quote_postcheck_enforce_mode: str = "soft"
    artifact_enforce_mode: str = "soft"
    min_recoverable_tokens: int = 8
    min_quote_chars: int = 25
    min_quote_content_tokens: int = 5
    static_safe_template: str = "Не удалось сформировать корректный анализ по данной новости."
    static_insufficient_template: str = "Недостаточно данных для анализа."
    trivial_quote_stoplist: list[str] = Field(default_factory=list)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    yellow_output_block_patterns: list[str] = Field(default_factory=list)
    sport_intra_negatives: list[str] = Field(default_factory=list)
    skip_templates: dict[str, str] = Field(default_factory=dict)
    trial50_hotfixes: dict[str, bool] = Field(default_factory=dict)
    fallback_templates: list[str] = Field(default_factory=list)
    min_meaningful_chars: int = 40
    answer_body_cleanup_enabled: bool = True
    integrity_check_enabled: bool = True
    integrity_enforce_mode: str = "soft"
    short_disclaimer: str = (
        "Ответ сгенерирован ИИ в образовательных целях "
        "(симуляция на основе трудов В.И. Ленина) и не является призывом к действию."
    )


def default_quality_postcheck_path(base_dir: Path) -> Path:
    return base_dir / "config" / "quality_postcheck.yaml"


def load_quality_postcheck_config(path: Path) -> QualityPostcheckConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("quality_postcheck", payload)
    return QualityPostcheckConfig.model_validate(section)


@lru_cache(maxsize=4)
def get_quality_postcheck_config(path_str: str) -> QualityPostcheckConfig:
    return load_quality_postcheck_config(path=Path(path_str))

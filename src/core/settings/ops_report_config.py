"""Loader for admin ops digest settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
import yaml


FetchNotify = Literal["never", "new_only"]
IdleDigest = Literal["short", "full"]


class OpsReportConfig(BaseModel):
    interval_seconds: int = 1800
    fetch_notify: FetchNotify = "new_only"
    top_reasons: int = 3
    idle_digest: IdleDigest = "short"
    max_latency_samples: int = 50


def default_ops_report_path(base_dir: Path) -> Path:
    return base_dir / "config" / "ops_report.yaml"


def load_ops_report_config(path: Path) -> OpsReportConfig:
    if not path.is_file():
        return OpsReportConfig()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("ops_report", payload)
    return OpsReportConfig.model_validate(section)


@lru_cache(maxsize=4)
def get_ops_report_config(path_str: str) -> OpsReportConfig:
    return load_ops_report_config(path=Path(path_str))

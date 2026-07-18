"""Typed registry for legacy RAG components pending archive/removal."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class LegacyComponent(BaseModel):
    path: str
    category: str
    status: str
    action: str
    note: str = ""


class LegacyRegistry(BaseModel):
    policy_version: str = "1.0.0"
    components: list[LegacyComponent] = Field(default_factory=list)


def load_legacy_registry(path: Path) -> LegacyRegistry:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("legacy_rag_registry", payload)
    return LegacyRegistry.model_validate(section)

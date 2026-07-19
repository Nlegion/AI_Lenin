"""NewsGuard-incident fallback recommendation hook."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.settings.generation_config import GenerationConfig, PersonaModel


def count_recent_high_risk_incidents(audit_log_path: Path, window_events: int) -> int:
    if not audit_log_path.exists():
        return 0
    lines = audit_log_path.read_text(encoding="utf-8").splitlines()[-window_events:]
    count = 0
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("high_risk"):
            count += 1
    return count


def recommend_persona_model(config: GenerationConfig, base_dir: Path) -> PersonaModel:
    """Return recommended persona model; auto-switch only when fallback.enabled."""
    active = config.persona_model
    fallback = config.safety.fallback
    if not fallback.enabled:
        return active
    configured = Path(fallback.audit_log_path)
    audit_path = configured.resolve() if configured.is_absolute() else (base_dir / configured).resolve()
    incidents = count_recent_high_risk_incidents(
        audit_log_path=audit_path,
        window_events=fallback.window_events,
    )
    if incidents >= fallback.incident_threshold:
        return fallback.target_persona_model
    return active

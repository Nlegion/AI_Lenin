"""Incident-audit hook for generation persona selection.

With a single runtime persona (base_strong / GigaChat3), recommendations
never switch backends; the audit counter remains for observability.
"""

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
    """Return active persona; no alternate backend remains after Saiga removal."""
    if config.safety.fallback.enabled:
        # Keep reading the audit path so ops can still enable counting without a switch.
        configured = Path(config.safety.fallback.audit_log_path)
        audit_path = configured.resolve() if configured.is_absolute() else (base_dir / configured).resolve()
        _ = count_recent_high_risk_incidents(
            audit_log_path=audit_path,
            window_events=config.safety.fallback.window_events,
        )
    return config.persona_model

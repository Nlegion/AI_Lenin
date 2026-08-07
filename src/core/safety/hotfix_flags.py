"""Trial50 hotfix feature flags (masters + per-rule children)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "safety_hotfixes_enabled": True,
    "generation_hotfixes_enabled": True,
    "drone_deny_enabled": True,
    "combat_adjacent_softpass_block": True,
    "sport_token_bound_enabled": True,
    "fio_carveout_enabled": True,
    "loop_strip_enabled": True,
    "encoding_scrubber_enabled": True,
    "disclaimer_footer_enabled": True,
}

_SAFETY_CHILDREN = frozenset(
    {
        "drone_deny_enabled",
        "combat_adjacent_softpass_block",
        "sport_token_bound_enabled",
        "fio_carveout_enabled",
    }
)
_GEN_CHILDREN = frozenset(
    {
        "loop_strip_enabled",
        "encoding_scrubber_enabled",
        "disclaimer_footer_enabled",
    }
)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "quality_postcheck.yaml"


@lru_cache(maxsize=4)
def _load_flags(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        return dict(_DEFAULTS)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("quality_postcheck", payload)
    hotfixes = section.get("trial50_hotfixes") or {}
    merged = dict(_DEFAULTS)
    merged.update({k: bool(v) for k, v in hotfixes.items() if k in _DEFAULTS})
    return merged


def reload_hotfix_flags() -> None:
    _load_flags.cache_clear()


def safety_flag_enabled(name: str) -> bool:
    flags = _load_flags(str(_config_path()))
    if not flags.get("safety_hotfixes_enabled", True):
        return False
    if name in _SAFETY_CHILDREN:
        return bool(flags.get(name, True))
    return bool(flags.get(name, True))


def generation_flag_enabled(name: str) -> bool:
    flags = _load_flags(str(_config_path()))
    if not flags.get("generation_hotfixes_enabled", True):
        return False
    if name in _GEN_CHILDREN:
        return bool(flags.get(name, True))
    return bool(flags.get(name, True))

"""Load anti-cliché YAML config."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_LEXICON = frozenset(
    {
        "революция",
        "эксплуатация",
        "пролетариат",
        "буржуазия",
        "классовая",
        "империализм",
        "диктатура",
    }
)
_DEFAULT_ANCHORS = (
    "как отмечает",
    "по данным",
    "согласно",
    "цитирует",
    "как писал",
    "в псс",
)


@dataclass(frozen=True)
class AntiClicheConfig:
    mode: str = "warn_only"
    min_r1_jaccard: float = 0.02
    lexicon_density_min_hits: int = 3
    lexicon: frozenset[str] = field(default_factory=lambda: _DEFAULT_LEXICON)
    quote_anchor_phrases: tuple[str, ...] = _DEFAULT_ANCHORS

    @property
    def warn_only(self) -> bool:
        return self.mode != "block"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=4)
def load_anti_cliche_config(path: str | None = None) -> AntiClicheConfig:
    config_path = Path(path) if path else _repo_root() / "config" / "anti_cliche.yaml"
    if not config_path.is_file():
        return AntiClicheConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lexicon_raw = payload.get("lexicon") or list(_DEFAULT_LEXICON)
    anchors_raw = payload.get("quote_anchor_phrases") or list(_DEFAULT_ANCHORS)
    return AntiClicheConfig(
        mode=str(payload.get("mode", "warn_only")),
        min_r1_jaccard=float(payload.get("min_r1_jaccard", 0.02)),
        lexicon_density_min_hits=int(payload.get("lexicon_density_min_hits", 3)),
        lexicon=frozenset(str(item).casefold() for item in lexicon_raw),
        quote_anchor_phrases=tuple(str(item).casefold() for item in anchors_raw),
    )

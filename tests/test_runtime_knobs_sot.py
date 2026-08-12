from __future__ import annotations

from pathlib import Path

from src.core.settings.runtime_knobs import load_reasoning_config_with_generation_sot


def test_reasoning_config_prefers_generation_sot() -> None:
    cfg = load_reasoning_config_with_generation_sot(base_dir=Path("."))
    # generation.yaml base_strong: temperature 0.4, max_tokens 512, ctx 4096
    assert cfg.temperature == 0.4
    assert cfg.max_tokens_out == 512
    assert cfg.ctx_size == 4096

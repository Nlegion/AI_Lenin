from __future__ import annotations

from src.core.generation.output_artifacts import apply_artifact_pass
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig


def test_artifact_pass_strips_chatml_and_orchestrator_labels() -> None:
    cfg = QualityPostcheckConfig()
    cfg.loop_fix_enabled = True
    text = (
        "<|im_start|>assistant Факт: Данные из R1 подтверждают событие. "
        "[multi-stance] [multi-stance] Механизм: ... Вывод: ... <|im_end|>"
    )
    result = apply_artifact_pass(text=text, config=cfg, combat_sensitive=False)
    assert "<|im_start|>" not in result.text
    assert "<|im_end|>" not in result.text
    assert "strip:chatml_token" in result.codes
    assert "strip:multi_stance_repeat" in result.codes
    assert "strip:orchestrator_label" in result.codes

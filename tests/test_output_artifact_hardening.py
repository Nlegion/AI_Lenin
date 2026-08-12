from __future__ import annotations

from src.core.generation.output_artifacts import apply_artifact_pass, final_public_scrub
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
    assert "[multi-stance]" not in result.text
    assert "R1" not in result.text
    assert "strip:chatml_token" in result.codes
    assert "strip:multi_stance" in result.codes
    assert "strip:orchestrator_label" in result.codes


def test_final_public_scrub_removes_prompt_echoes() -> None:
    text = (
        "Факт: событие произошло. Механизм: анализ. Вывод: итог. "
        "[multi-stance] (пусто) Доказательная база (не выдумывай вне этих блоков): "
        "## [1] ( 48. текст) контекст — согласие"
    )
    cleaned, codes = final_public_scrub(text)
    assert "[multi-stance]" not in cleaned
    assert "(пусто)" not in cleaned
    assert "Доказательная база" not in cleaned
    assert "[1]" not in cleaned
    assert "контекст — согласие" not in cleaned
    assert codes

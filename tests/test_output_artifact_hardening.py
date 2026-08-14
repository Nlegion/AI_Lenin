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


def test_artifact_pass_strips_inline_stance_and_instruction() -> None:
    cfg = QualityPostcheckConfig()
    text = (
        "Факт: событие. Механизм: анализ. Вывод: итог. "
        "— Ленин (core_approval) — Ленин (core_criticism) "
        "Запрещено выдумывать цитаты, кавычки и том/стр."
    )
    result = apply_artifact_pass(text=text, config=cfg)
    assert "Ленин (core_approval)" not in result.text
    assert "Запрещено выдумывать" not in result.text
    assert "Факт: событие" in result.text


def test_final_public_scrub_strips_mesto_and_obezlicheno() -> None:
    text = (
        "Факт: чиновник из «[место]» заявил о реформе. "
        "Механизм: [обезличено] влияет на бюджет. "
        "Вывод: нужен контроль,, классов."
    )
    cleaned, codes = final_public_scrub(text)
    assert "«[место]»" not in cleaned
    assert "[место]" not in cleaned
    assert "[обезличено]" not in cleaned
    assert "strip:mesto_marker" in codes
    assert "«»" not in cleaned
    assert ",," not in cleaned
    assert "чиновник из" in cleaned
    assert "заявил о реформе" in cleaned


def test_final_public_scrub_after_newsguard_insert() -> None:
    """Markers inserted after body cleanup are removed on the public path."""
    from src.core.generation.answer_body_cleanup import cleanup_answer_body

    cfg = QualityPostcheckConfig()
    body = cleanup_answer_body(
        text="Факт: Иванов сообщил. Механизм: анализ. Вывод: итог.",
        config=cfg,
    )
    assert "«[место]»" not in body.text
    # Simulate NewsGuard PII redact after body cleanup.
    guarded = body.text.replace("Иванов", "«[место]»")
    assert "«[место]»" in guarded
    cleaned, codes = final_public_scrub(guarded)
    assert "«[место]»" not in cleaned
    assert "Иванов" not in cleaned  # PII not restored
    assert "strip:mesto_marker" in codes
    # Reasoning publish path uses the same scrub helper.
    assert final_public_scrub(guarded)[0] == cleaned
